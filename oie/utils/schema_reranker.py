# oie/utils/schema_reranker.py
import logging
from dataclasses import dataclass
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SchemaRerankerConfig:
    reranker_model_name_or_path: str
    batch_size: int = 32
    max_length: int = 4096
    device: str = "cuda"  # or "cpu"
    torch_dtype: str = "auto"  # "auto" or "float16" etc.
    show_progress: bool = False
    progress_desc: str = "SchemaReranker"
    tqdm_position: int = 1


class SchemaReranker:
    """
    Cross-encoder reranker: score(query, candidate_text) for each candidate.
    Uses sequence classification logits as scores (no softmax needed for ranking).
    """
    def __init__(self, cfg: SchemaRerankerConfig):
        self.cfg = cfg

        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[SchemaReranker] CUDA not available. Falling back to CPU.")
            self.cfg.device = "cpu"

        logger.info(f"[Step3_SchemaReranker] reranker model: {self.cfg.reranker_model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.reranker_model_name_or_path)

        dtype = None
        if self.cfg.torch_dtype and self.cfg.torch_dtype != "auto":
            dtype = getattr(torch, self.cfg.torch_dtype, None)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.reranker_model_name_or_path,
            torch_dtype=dtype,
        )
        self.model.eval()
        self.model.to(self.cfg.device)

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

    # @torch.inference_mode()
    # def score(self, query_text: str, candidate_texts: List[str]) -> List[float]:
    #     """
    #     Returns list of logits (float), aligned with candidate_texts.
    #     """
    #     if not candidate_texts:
    #         return []

    #     scores: List[float] = []
    #     bs = self.cfg.batch_size

    #     for i in range(0, len(candidate_texts), bs):
    #         batch_cands = candidate_texts[i:i + bs]

    #         # Pair encoding: (query, candidate)
    #         tok = self.tokenizer(
    #             [query_text] * len(batch_cands),
    #             batch_cands,
    #             padding=True,
    #             truncation=True,
    #             max_length=self.cfg.max_length,
    #             return_tensors="pt",
    #         ).to(self.cfg.device)

    #         out = self.model(**tok, return_dict=True)
    #         logits = out.logits  # (B, 1) or (B,)
    #         if logits.dim() == 2 and logits.size(-1) == 1:
    #             logits = logits.squeeze(-1)

    #         scores.extend([float(x) for x in logits.detach().cpu().tolist()])

    #     return scores
    @torch.inference_mode()
    def score(self, query_text: str, candidate_texts: List[str]) -> List[float]:
        if not candidate_texts:
            return []

        scores: List[float] = []
        bs = self.cfg.batch_size

        rng = range(0, len(candidate_texts), bs)
        if self.cfg.show_progress and len(candidate_texts) > bs:
            rng = tqdm(
                rng,
                total=(len(candidate_texts) + bs - 1) // bs,
                desc=self.cfg.progress_desc + ": rerank",
                leave=False,
                position=self.cfg.tqdm_position,
            )

        for i in rng:
            batch_cands = candidate_texts[i:i + bs]
            tok = self.tokenizer(
                [query_text] * len(batch_cands),
                batch_cands,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.cfg.device)

            out = self.model(**tok, return_dict=True)
            logits = out.logits
            if logits.dim() == 2 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)

            scores.extend([float(x) for x in logits.detach().cpu().tolist()])

        return scores