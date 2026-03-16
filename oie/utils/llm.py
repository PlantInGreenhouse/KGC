import json
import re
import ast
from typing import List, Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def apply_chat_template_or_fallback(tokenizer: AutoTokenizer, messages: List[dict]) -> str:
    """
    Pure function: returns a string prompt.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback: simple role-content concatenation
    return "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages]) + "\nASSISTANT:"


def generate_completion_transformers(
    messages: List[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    answer_prepend: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    top_p: float = 1.0,
) -> str:
    """
    Stateless generation helper.
    - No model/tokenizer caching here.
    - Designed to be called repeatedly without VRAM growth.
    """

    prompt = apply_chat_template_or_fallback(tokenizer, messages)
    if answer_prepend:
        prompt = prompt + "\n" + answer_prepend

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0.0

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # decode only generated tokens (after prompt length)
    gen = output_ids[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text

def _clean_cell(s: str) -> str:
    if not isinstance(s, str):
        return s
    t = s.strip()

    t = t.rstrip(" ,")
    t = t.strip()

    if (len(t) >= 2) and ((t[0] == t[-1]) and t[0] in ("'", '"')):
        t = t[1:-1].strip()

    if (len(t) >= 2) and ((t[0] == t[-1]) and t[0] in ("'", '"')):
        t = t[1:-1].strip()

    t = t.replace("]", "").replace("[", "").strip()

    return t


def _try_parse_json_list(text: str) -> Optional[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    chunk = text[start:end + 1].strip()

    try:
        return json.loads(chunk)
    except Exception:
        pass
    
    try:
        obj = ast.literal_eval(chunk)
        return obj
    except Exception:
        return None


def parse_raw_triplets(text: str) -> List[List[str]]:
    obj = _try_parse_json_list(text)

    if isinstance(obj, list):
        triples = []
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) == 3:
                s, r, o = item
                if all(isinstance(x, str) for x in (s, r, o)):
                    s2, r2, o2 = _clean_cell(s), _clean_cell(r), _clean_cell(o)
                    # 빈 값/언더스코어 필터
                    if all(x and x != "_" for x in (s2, r2, o2)):
                        triples.append([s2, r2, o2])
        if triples:
            return triples

    # 이하 기존 fallback regex 파서도 "clean"을 적용하도록 수정
    triples: List[List[str]] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    pat_paren = re.compile(r"^\(\s*(.*?)\s*;\s*(.*?)\s*;\s*(.*?)\s*\)\s*$")
    pat_pipe = re.compile(r"^(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*$")
    pat_bracket = re.compile(r"^\[\s*(.*?)\s*,\s*(.*?)\s*,\s*(.*?)\s*\]\s*$")

    for ln in lines:
        m = pat_paren.match(ln) or pat_pipe.match(ln) or pat_bracket.match(ln)
        if not m:
            continue
        s, r, o = _clean_cell(m.group(1)), _clean_cell(m.group(2)), _clean_cell(m.group(3))
        if all(x and x != "_" for x in (s, r, o)):
            triples.append([s, r, o])

    return triples