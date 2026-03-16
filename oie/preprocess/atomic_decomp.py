import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are an expert at transforming ONE input sentence into atomic facts.

Hard constraints:
- Output ONLY atomic facts. No explanations, no notes, no preambles, no extra text.
- Output format: one fact per line, exactly: S1 → ...
- Each line MUST start with S<number> → (e.g., S1 →, S2 →).
- Exactly ONE claim per line (single-claim).
- No pronouns (he/she/it/they/this/that). Use explicit names.
- Copy proper nouns and numbers/dates EXACTLY from the input (do not change spelling or digits).
- Do NOT infer unstated facts (e.g., "born in X" does NOT imply "lives in X").
- Do NOT mention "knowledge base" or "No new facts". Never output such sentences.
- Do NOT use the symbol "→" inside the sentence text; use it ONLY as the delimiter after S<number>.
- If the sentence contains facts, list ALL extractable atomic facts from the sentence, then stop.
""".strip()

EX1_IN = 'Gabriel Obadin Aikhena (born 9 May 1986 in Iseyin) is a Singaporean-Nigerian footballer who plays for Nay Pyi Taw F.C.'
EX1_OUT = "\n".join([
    "S1 → Gabriel Obadin Aikhena is a footballer.",
    "S2 → Gabriel Obadin Aikhena plays for Nay Pyi Taw F.C.",
    "S3 → Gabriel Obadin Aikhena was born on 9 May 1986.",
    "S4 → Gabriel Obadin Aikhena was born in Iseyin.",
    "S5 → Gabriel Obadin Aikhena is Singaporean.",
    "S6 → Gabriel Obadin Aikhena is Nigerian.",
])

EX2_IN = 'Olaf Kapagiannidis (born 11 June 1969 in Berlin-Spandau) is a former professional German footballer.'
EX2_OUT = "\n".join([
    "S1 → Olaf Kapagiannidis is a footballer.",
    "S2 → Olaf Kapagiannidis was born on 11 June 1969.",
    "S3 → Olaf Kapagiannidis was born in Berlin-Spandau.",
    "S4 → Olaf Kapagiannidis is a former professional player.",
    "S5 → Olaf Kapagiannidis is German.",
])

USER_TASK_TMPL = """
Convert the sentence into atomic facts.

Checklist (do internally, do NOT output):
- Split parenthetical/appositive info (born/died, places, titles).
- Split compound attributes (e.g., "Nigerian football midfielder" -> nationality + occupation + position).
- Split coordination ("and", commas) into separate facts when they represent distinct claims.
- For "X or Y" naming, output alias ONLY if it is clearly an alias/name variant.

Sentence: "{sentence}"

Output (ONLY S-lines):
""".strip()

S_LINE_RE = re.compile(r"^\s*(?:•\s*)?(S\d+)\s*→\s*(.+?)\s*$")


class RepetitionStopper(StoppingCriteria):
    def __init__(self, window: int = 48, min_length: int = 120):
        self.window = window
        self.min_length = min_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        if seq.numel() < self.min_length or seq.numel() < 2 * self.window:
            return False
        return torch.equal(seq[-self.window:], seq[-2*self.window:-self.window])


def _digits_in(text: str) -> List[str]:
    return re.findall(r"\d+", text)


def parse_s_lines(raw: str) -> List[str]:
    atoms: List[str] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        m = S_LINE_RE.match(ln)
        if m:
            atoms.append(m.group(2).strip())
    return atoms


def validate_atoms(atoms: List[str], sentence: str) -> bool:
    """Fail-fast validation to catch obvious degradation/hallucination."""
    if not atoms:
        return False

    # reject forbidden meta outputs
    joined = " ".join(atoms).lower()
    if "knowledge base" in joined or "no new facts" in joined:
        return False

    # Ensure digits in output are a subset of digits in input (prevents 1975->1977 type drift)
    in_digits = set(_digits_in(sentence))
    out_digits = _digits_in(" ".join(atoms))
    for d in out_digits:
        if d not in in_digits:
            return False

    # Disallow delimiter symbol inside atom text
    if any("→" in a for a in atoms):
        return False

    return True


def _get_eos_ids(tokenizer) -> List[int]:
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(tokenizer.eos_token_id)
    for tok in ["<|eot_id|>", "<|end_of_turn|>"]:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tokenizer.unk_token_id is not None and tid != tokenizer.unk_token_id:
            eos_ids.append(tid)
    # unique preserve order
    return list(dict.fromkeys(eos_ids))


def _build_messages(sentence: str, strict: bool = False) -> List[Dict[str, str]]:
    user = USER_TASK_TMPL.format(sentence=sentence)
    if strict:
        user += "\nSTRICT REMINDER: Output MUST be ONLY lines starting with 'S<number> → '. No other text.\n"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TASK_TMPL.format(sentence=EX1_IN)},
        {"role": "assistant", "content": EX1_OUT},
        {"role": "user", "content": USER_TASK_TMPL.format(sentence=EX2_IN)},
        {"role": "assistant", "content": EX2_OUT},
        {"role": "user", "content": user},
    ]


@torch.inference_mode()
def generate_raw(model, tokenizer, sentence: str, max_new_tokens: int, strict: bool) -> str:
    messages = _build_messages(sentence, strict=strict)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)

    eos_ids = _get_eos_ids(tokenizer)
    stopping = StoppingCriteriaList([RepetitionStopper(window=48, min_length=120)])

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        # IMPORTANT: no_repeat_ngram_size 제거 (atomic에 치명적)
        repetition_penalty=1.05,  # 약하게만
        eos_token_id=eos_ids if eos_ids else tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        stopping_criteria=stopping,
    )

    gen_ids = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def generate_atomic_with_retry(model, tokenizer, sentence: str, base_tokens: int = 256, retries: int = 2) -> Tuple[str, List[str]]:
    last_raw = ""
    last_atoms: List[str] = []

    # 문장 길이에 따라 토큰 예산 약간 가변
    # (너무 짧으면 birth/death + roles 등을 끝까지 못 씀)
    budget0 = max(base_tokens, min(384, 64 + len(sentence) // 2))

    for attempt in range(retries + 1):
        strict = (attempt > 0)
        budget = budget0 if attempt == 0 else min(budget0 + 128, 512)

        raw = generate_raw(model, tokenizer, sentence, max_new_tokens=budget, strict=strict)
        atoms = parse_s_lines(raw)

        last_raw, last_atoms = raw, atoms

        if validate_atoms(atoms, sentence):
            return raw, atoms

    return last_raw, last_atoms  # 실패 시 raw 저장용


@dataclass
class AtomicDecompConfig:
    model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
    base_tokens: int = 256
    retries: int = 2
    loglevel: int = logging.INFO


class AtomicDecomposer:
    def __init__(self, cfg: AtomicDecompConfig):
        self.cfg = cfg
        logger.setLevel(cfg.loglevel)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device

        tok = AutoTokenizer.from_pretrained(cfg.model_id, use_fast=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self.tokenizer = tok

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to(device)
        model.eval()
        self.model = model

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

    def run_txt(self, input_txt: str, out_jsonl: str) -> None:
        input_path = Path(input_txt)
        if not input_path.exists():
            raise FileNotFoundError(f"Input txt not found: {input_path.resolve()}")

        sentences = [ln.strip() for ln in input_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not sentences:
            raise ValueError(f"{input_txt} is empty.")

        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", encoding="utf-8") as f:
            for idx, sent in tqdm(list(enumerate(sentences, start=1)), total=len(sentences),
                                 desc="Atomic decomposition", unit="sent", dynamic_ncols=True):
                raw, atoms = generate_atomic_with_retry(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    sentence=sent,
                    base_tokens=self.cfg.base_tokens,
                    retries=self.cfg.retries,
                )

                rec: Dict[str, Any] = {"index": idx, "input": sent, "atomic": atoms}
                if not validate_atoms(atoms, sent):
                    rec["raw_model_output"] = raw
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self.close()