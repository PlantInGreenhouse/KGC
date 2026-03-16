import argparse
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_RELATIONS = {
    "ContentScope",
    "MethodEnablement",
    "Provenance",
    "ContextLocale",
    "TemporalGate",
    "Rationale",
    "None",
}

SYSTEM_PROMPT = """### Role
You are an Engineer using a RST-inspired relation taxonomy.
Your task is to classify the relation between two Atomic Facts (AF1, AF2) given the Source Context.

### Directionality Constraint (IMPORTANT)
- AF1 is ALWAYS the Nucleus (main fact).
- AF2 is ALWAYS the Satellite (supporting fact that modifies/explains AF1).
- Choose the single best relation type that describes how AF2 supports AF1.
- Use ONLY explicit evidence from Source Context / AF1 / AF2. Never use world knowledge.
- If evidence is weak/ambiguous OR multiple labels plausible without decisive markers, choose "None".

### Relation Taxonomy (Fixed 6 + None) with RST Mapping
Select exactly ONE relation from the list below.

1) ContentScope  (RST: Elaboration, Summary)  [What]
Function:
- AF2 adjusts the content scope of AF1 by either:
  (a) ELABORATION: adding clarifying/detailing information that makes AF1 more specific, or
  (b) SUMMARY: compressing AF1 into a higher-level gist.
When to use:
- AF2 answers: "What does AF1 mean / what is AF1 about (in more detail or in brief)?"

2) MethodEnablement  (RST: Manner-Means, Enablement)  [How]
Function:
- AF2 provides HOW AF1 is achieved/performed/realized:
  (a) MANNER/MEANS: method, instrument, procedure, mechanism
  (b) ENABLEMENT: a facilitating action/condition that makes AF1 possible
When to use:
- AF2 answers: "How does AF1 happen / how is AF1 done / how is AF1 made possible?"

3) Provenance  (RST: Attribution)  [Who]
Function:
- AF2 explicitly attributes AF1 to a source/author/creator/owner/issuer.
When to use:
- AF2 answers: "Who said/wrote/created/reported AF1?" / "According to whom?"

4) ContextLocale  (RST: Background — locative/inclusion frame)  [Where]
Function:
- AF2 provides a WHERE/INCLUSION background frame that situates AF1 spatially or organizationally.
- This is not a mere location attribute; it must frame/locate AF1 (background setting).
When to use:
- AF2 answers: "Where is AF1 located / within what larger entity does AF1 hold?"

5) TemporalGate  (RST: Temporal, Condition)  [When]
Function:
- AF2 provides a time anchor or temporal/conditional frame for interpreting AF1.
- Includes explicit time expressions (date/year/period) and explicit temporal order/condition markers.
When to use:
- AF2 answers: "When does AF1 happen / under what time/if-frame does AF1 hold?"

6) Rationale  (RST: Cause, Explanation)  [Why]
Function:
- AF2 provides an explicit cause/reason/explanation/evidence for AF1, OR an explicit consequence/result of AF1.
When to use:
- AF2 answers: "Why did AF1 happen?" or "What happened because of AF1?"

7) None
Use when:
- AF2 does not clearly support/modify/frame/attribute AF1 under the above definitions,
- OR cues are weak,
- OR multiple labels plausible with no decisive evidence.

### Confidence (IMPORTANT)
Return a numeric confidence in [0.0, 1.0] reflecting how strongly the given text supports your chosen relation.
- 0.90–1.00: explicit evidence/cues; very clear.
- 0.60–0.89: plausible with clear support but not perfectly explicit.
- 0.30–0.59: weak support; borderline (if truly borderline, prefer "None").
- 0.00–0.29: essentially unsupported (should be "None").

### Output Format (JSON only)
Return ONLY a valid JSON object with keys:
- "relation": one of [ContentScope, MethodEnablement, Provenance, ContextLocale, TemporalGate, Rationale, None]
- "confidence": float in [0.0, 1.0]
- "reason": short justification (1–2 sentences), grounded in the given text only.
"""

# ------------------------------------
# 2) Few-shot as chat history (FICL)
# ------------------------------------
FICL_TURNS: List[Dict[str, str]] = [
    {
        "role": "user",
        "content": """### Source Context
"Count Basie was an American jazz pianist."

AF1 (Nucleus): "Count Basie is a person."
AF2 (Satellite): "Count Basie is a pianist."

Return JSON:"""
    },
    {
        "role": "assistant",
        "content": """{"relation":"ContentScope","confidence":0.92,"reason":"AF2 elaborates the essential professional type of the entity in AF1."}"""
    },

    {
        "role": "user",
        "content": """### Source Context
"The model was trained by using reinforcement learning."

AF1 (Nucleus): "The model was trained."
AF2 (Satellite): "It was trained by using reinforcement learning."

Return JSON:"""
    },
    {
        "role": "assistant",
        "content": """{"relation":"MethodEnablement","confidence":0.90,"reason":"AF2 states the explicit method/means for AF1 via the phrase 'by using'."}"""
    },

    {
        "role": "user",
        "content": """### Source Context
"Basie Rides Again! is an album by Count Basie."

AF1 (Nucleus): "Basie Rides Again! is an album."
AF2 (Satellite): "It is by Count Basie."

Return JSON:"""
    },
    {
        "role": "assistant",
        "content": """{"relation":"Provenance","confidence":0.90,"reason":"AF2 attributes AF1 to a creator/artist indicated by 'by Count Basie' in the context."}"""
    },

    {
        "role": "user",
        "content": """### Source Context
"It was recorded in 1952."

AF1 (Nucleus): "It was recorded."
AF2 (Satellite): "It was recorded in 1952."

Return JSON:"""
    },
    {
        "role": "assistant",
        "content": """{"relation":"TemporalGate","confidence":0.93,"reason":"AF2 provides an explicit time anchor ('in 1952') that frames AF1."}"""
    },

    {
        "role": "user",
        "content": """### Source Context
"The match was postponed because of heavy rain."

AF1 (Nucleus): "The match was postponed."
AF2 (Satellite): "It was postponed because of heavy rain."

Return JSON:"""
    },
    {
        "role": "assistant",
        "content": """{"relation":"Rationale","confidence":0.95,"reason":"AF2 gives an explicit cause for AF1 using the cue 'because of'."}"""
    },

    {
        "role": "user",
        "content": """### Source Context
"He was born in 1946. The album is jazz."

AF1 (Nucleus): "He was born in 1946."
AF2 (Satellite): "The album is jazz."

Return JSON:"""
    },
    {
        "role": "assistant",
        "content": """{"relation":"None","confidence":0.90,"reason":"AF2 is unrelated to AF1 and does not support or explain it in the context."}"""
    },
]

# ----------------------------
# 3) USER TEMPLATE (variant)
# ----------------------------
USER_PROMPT_TEMPLATE = """### Source Context
"{RAW_TEXT}"

AF1 (Nucleus): "{AF1_TEXT}"
AF2 (Satellite): "{AF2_TEXT}"

Return JSON:
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def compute_total_pairs(rows: List[Dict[str, Any]]) -> int:
    total = 0
    for r in rows:
        afs = r.get("atomic", [])
        n = len(afs)
        if n >= 2:
            total += n * (n - 1)
    return total


def build_pairs(atomic: List[str]) -> List[Tuple[int, int]]:
    pairs = []
    n = len(atomic)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pairs.append((i, j))
    return pairs


def build_user_prompt(raw_text: str, af1: str, af2: str) -> str:
    return USER_PROMPT_TEMPLATE.format(RAW_TEXT=raw_text, AF1_TEXT=af1, AF2_TEXT=af2)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_obj(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None

    cand = m.group(0).strip()
    cand = re.sub(r",\s*([}\]])", r"\1", cand)
    cand = cand.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    try:
        return json.loads(cand)
    except Exception:
        return None


def _to_float_confidence(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    # allow "0.87", "87%", "0,87"
    s = s.replace(",", ".")
    m = re.search(r"(\d+(\.\d+)?)", s)
    if not m:
        return None
    val = float(m.group(1))
    if "%" in s and val > 1.0:
        val = val / 100.0
    return val


def normalize_relation(obj: Dict[str, Any]) -> Dict[str, Any]:
    rel = str(obj.get("relation", "None")).strip()
    reason = str(obj.get("reason", "")).strip()

    if rel not in ALLOWED_RELATIONS:
        rel = "None"

    conf_raw = obj.get("confidence", None)
    conf = _to_float_confidence(conf_raw)
    if conf is None:
        conf = 0.0

    # clamp
    if conf < 0.0:
        conf = 0.0
    if conf > 1.0:
        conf = 1.0

    return {"relation": rel, "confidence": conf, "reason": reason}


@dataclass
class WeakLinkConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    batch_size: int = 8
    max_new_tokens: int = 4096
    max_length: int = 8192
    no_ficl: bool = False
    dtype: str = "bfloat16"  # bfloat16 권장
    device_map: str = "cuda"
    loglevel: int = logging.INFO


class WeakLinkTyper:
    def __init__(self, cfg: WeakLinkConfig):
        self.cfg = cfg
        logger.setLevel(cfg.loglevel)

        if cfg.dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif cfg.dtype == "float16":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        self.tokenizer = tok

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch_dtype,
            device_map=cfg.device_map,
        )
        model.eval()
        model.config.pad_token_id = tok.pad_token_id
        self.model = model

        shared = [{"role": "system", "content": SYSTEM_PROMPT}]
        if not cfg.no_ficl:
            shared += FICL_TURNS
        self.shared_turns = shared

    def close(self):
        try:
            del self.model
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run_jsonl(self, in_atomic_jsonl: str, out_rel_jsonl: str) -> None:
        rows = load_jsonl(in_atomic_jsonl)
        total_pairs = compute_total_pairs(rows)

        out_path = Path(out_rel_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_rel_jsonl, "w", encoding="utf-8") as out_f:
            pbar = tqdm(total=total_pairs, desc="Weak link typing", dynamic_ncols=True)

            for r in rows:
                idx = r.get("index")
                raw_text = r.get("input", "")
                atomic = r.get("atomic", [])
                n = len(atomic)

                pairs = build_pairs(atomic)
                relations: List[Dict[str, Any]] = []

                batch_texts: List[str] = []
                batch_meta: List[Tuple[int, int]] = []

                def flush_batch():
                    nonlocal batch_texts, batch_meta, relations
                    if not batch_texts:
                        return

                    inputs = self.tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self.cfg.max_length,
                    )
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) for k, v in inputs.items()}

                    with torch.no_grad():
                        generated = self.model.generate(
                            **inputs,
                            max_new_tokens=self.cfg.max_new_tokens,
                            do_sample=False,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )

                    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

                    decoded = []
                    for seq, in_len in zip(generated, input_lengths):
                        gen_tokens = seq[in_len:]
                        decoded.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))

                    for (i, j), txt in zip(batch_meta, decoded):
                        obj = extract_json_obj(txt)
                        if obj is None:
                            relations.append({
                                "af1_id": i,
                                "af2_id": j,
                                "af1": atomic[i],
                                "af2": atomic[j],
                                "relation": "None",
                                "confidence": 0.0,
                                "reason": "",
                                "parse_error": True,
                                "raw_output": txt[-800:],
                            })
                        else:
                            norm = normalize_relation(obj)
                            relations.append({
                                "af1_id": i,
                                "af2_id": j,
                                "af1": atomic[i],
                                "af2": atomic[j],
                                "relation": norm["relation"],
                                "confidence": norm["confidence"],
                                "reason": norm["reason"],
                                "parse_error": False,
                            })

                    pbar.update(len(batch_texts))
                    batch_texts = []
                    batch_meta = []

                for (i, j) in pairs:
                    user_prompt = build_user_prompt(raw_text, atomic[i], atomic[j])
                    messages = self.shared_turns + [{"role": "user", "content": user_prompt}]
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    batch_texts.append(text)
                    batch_meta.append((i, j))

                    if len(batch_texts) >= self.cfg.batch_size:
                        flush_batch()

                flush_batch()

                out_record = {
                    "index": idx,
                    "input": raw_text,
                    "atomic": atomic,
                    "relations": relations,
                }
                out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")

            pbar.close()

        self.close()