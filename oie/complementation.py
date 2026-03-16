# oie/complementation.py
import json
import logging
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import llm

logger = logging.getLogger(__name__)

@dataclass
class ComplementationConfig:
    llm_name_or_path: str
    prompt_template_file_path: str
    few_shot_example_file_path: str
    loglevel: int = logging.INFO

class ComplementationEngine:
    _MAX_NEW_TOKENS = 4096
    _TEMPERATURE = 0.0
    _TOP_P = 1.0

    _DEVICE_MAP = "cuda"
    _TORCH_DTYPE = "auto"

    def __init__(self, cfg: ComplementationConfig):
        self.cfg = cfg
        logger.setLevel(cfg.loglevel)

        self.prompt_template = open(cfg.prompt_template_file_path, "r", encoding="utf-8").read()
        self.few_shot_examples = open(cfg.few_shot_example_file_path, "r", encoding="utf-8").read()

        logger.info(f"[Complementation Model]: {cfg.llm_name_or_path}")

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.llm_name_or_path,
            device_map=self._DEVICE_MAP,
            dtype=None if self._TORCH_DTYPE == "auto" else self._TORCH_DTYPE,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name_or_path)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, raw_sentence: str, existing_triples: List[List[str]], complementation_hints: str, few_shot_examples: str) -> str:
        existing_json = json.dumps(existing_triples or [], ensure_ascii=False)
        return self.prompt_template.format_map({
            "raw_sentence": raw_sentence.strip(),
            "existing_triples_json": existing_json,
            "complementation_hints": complementation_hints.strip(),
            "few_shot_examples": few_shot_examples.strip(),
        })

    def extract_new_triples(
        self,
        raw_sentence: str,
        existing_triples: List[List[str]],
        complementation_hints: str,
        few_shot_examples: str,
    ) -> List[List[str]]:
        prompt = self._build_prompt(raw_sentence, existing_triples, complementation_hints, few_shot_examples)
        # print(prompt)
        messages = [{"role": "user", "content": prompt}]

        completion = llm.generate_completion_transformers(
            messages=messages,
            model=self.model,
            tokenizer=self.tokenizer,
            answer_prepend="",
            max_new_tokens=self._MAX_NEW_TOKENS,
            temperature=self._TEMPERATURE,
            top_p=self._TOP_P,
        )

        triples = llm.parse_raw_triplets(completion)
        return triples

    def close(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass