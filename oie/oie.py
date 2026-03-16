import logging
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .utils import llm

logger = logging.getLogger(__name__)


@dataclass
class OIEConfig:
    oie_llm: str
    oie_prompt_template_file_path: str
    oie_few_shot_example_file_path: str
    loglevel: int = logging.INFO


class OIEEngine:
    """
    Owns model/tokenizer and prompt assets (single-load).
    Uses utils.llm for generation + parsing (stateless).
    """
    _MAX_NEW_TOKENS = 4096
    _TEMPERATURE = 0.0
    _TOP_P = 1.0

    _DEVICE_MAP = "cuda"
    _TORCH_DTYPE = "auto"

    def __init__(self, cfg: OIEConfig):
        self.cfg = cfg
        logger.setLevel(cfg.loglevel)

        self.prompt_template = open(cfg.oie_prompt_template_file_path, "r", encoding="utf-8").read()
        self.few_shot = open(cfg.oie_few_shot_example_file_path, "r", encoding="utf-8").read()

        logger.info(f"[OIE Model]: {cfg.oie_llm}")

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.oie_llm,
            device_map=self._DEVICE_MAP,
            torch_dtype=None if self._TORCH_DTYPE == "auto" else self._TORCH_DTYPE,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.oie_llm)

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _build_prompt(self, raw_sentence: str, atomic: List[str]) -> str:
        atomic_facts_str = "\n".join(
            [f"{i}. {af.strip()}" for i, af in enumerate(atomic, start=1)]
        )

        return self.prompt_template.format_map(
            {
                "few_shot_examples": self.few_shot,
                "raw_sentence": raw_sentence.strip(),
                "atomic_facts": atomic_facts_str,
            }
        )
    
    def extract_one(self, raw_sentence: str, atomic: List[str]) -> List[List[str]]:
        prompt = self._build_prompt(raw_sentence, atomic)
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

    def extract_batch(self, raw_sentence_list: List[str], atomic_list: List[List[str]]) -> List[List[List[str]]]:
        assert len(raw_sentence_list) == len(atomic_list)
        outputs = []
        for raw, atomic in zip(raw_sentence_list, atomic_list):
            outputs.append(self.extract_one(raw, atomic))
        return outputs

    def close(self):
        # Best-effort VRAM cleanup
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