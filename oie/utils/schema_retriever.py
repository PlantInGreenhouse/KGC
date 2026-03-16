# oie/utils/schema_retriever.py
import csv
import logging
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class SchemaItem:
    name: str
    definition: str
    text: str  # f"{name}: {definition}"


@dataclass
class Candidate:
    name: str
    definition: str
    score: float


@dataclass
class SchemaRetrieverConfig:
    schema_csv_path: str
    embed_model_name_or_path: str
    top_k: int = 5
    batch_size: int = 64
    max_length: int = 4096
    device: str = "cuda"  # or "cpu"
    normalize: bool = True  # L2 normalize embeddings for cosine
    # (선택) e5 계열이면 prefix가 성능에 도움
    query_prefix: str = ""      # e.g., "query: "
    schema_prefix: str = ""     # e.g., "passage: "
    show_progress: bool = False
    progress_desc: str = "SchemaRetriever"


class SchemaRetriever:
    def __init__(self, cfg: SchemaRetrieverConfig):
        self.cfg = cfg

        # device fallback
        if self.cfg.device == "cuda" and not torch.cuda.is_available():
            logger.warning("[SchemaRetriever] CUDA not available. Falling back to CPU.")
            self.cfg.device = "cpu"

        # auto prefix for e5 (if not set)
        mname = (self.cfg.embed_model_name_or_path or "").lower()
        if "e5" in mname:
            if not self.cfg.query_prefix:
                self.cfg.query_prefix = "query: "
            if not self.cfg.schema_prefix:
                self.cfg.schema_prefix = "passage: "

        self.schema_items: List[SchemaItem] = self._load_schema(self.cfg.schema_csv_path)

        logger.info(f"[Step3_SchemaRetriever] schema items: {len(self.schema_items)}")
        logger.info(f"[Step3_SchemaRetriever] embed model: {self.cfg.embed_model_name_or_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.embed_model_name_or_path)
        self.model = AutoModel.from_pretrained(self.cfg.embed_model_name_or_path)
        self.model.eval()
        self.model.to(self.cfg.device)

        # build schema embeddings once
        # with torch.inference_mode():
        #     schema_texts = [self.cfg.schema_prefix + it.text for it in self.schema_items]
        #     self.schema_emb = self._embed_texts(schema_texts)  # (N, D), on CPU
        #     if self.cfg.normalize:
        #         self.schema_emb = F.normalize(self.schema_emb, p=2, dim=1)
        with torch.inference_mode():
            schema_texts = [self.cfg.schema_prefix + it.text for it in self.schema_items]
            self.schema_emb = self._embed_texts(
                schema_texts,
                show_progress=self.cfg.show_progress,
                desc=self.cfg.progress_desc + ": embed schema",
            )
            if self.cfg.normalize:
                self.schema_emb = F.normalize(self.schema_emb, p=2, dim=1)

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

    def _load_schema(self, csv_path: str) -> List[SchemaItem]:
        items: List[SchemaItem] = []
        # utf-8-sig: BOM safe
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                name = (row[0] or "").strip()
                definition = (row[1] or "").strip()

                # header skip
                if name.lower() in ("name", "relation", "predicate") and definition.lower() in ("definition", "desc", "description"):
                    continue

                if not name or not definition:
                    continue
                text = f"{name}: {definition}"
                items.append(SchemaItem(name=name, definition=definition, text=text))

        if len(items) == 0:
            raise ValueError(f"Empty or invalid schema csv: {csv_path}")
        return items

    def _mean_pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # last_hidden_state: (B, T, H), attention_mask: (B, T)
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
        summed = (last_hidden_state * mask).sum(dim=1)                  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)                        # (B, 1)
        return summed / counts

    def _embed_texts(self, texts: List[str], show_progress: bool = False, desc: str = "") -> torch.Tensor:
        embs: List[torch.Tensor] = []
        bs = self.cfg.batch_size

        rng = range(0, len(texts), bs)
        if show_progress:
            rng = tqdm(rng, total=(len(texts) + bs - 1) // bs, desc=desc, leave=False)

        for i in rng:
            batch = texts[i:i + bs]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            ).to(self.cfg.device)

            out = self.model(**tok, return_dict=True)
            pooled = self._mean_pool(out.last_hidden_state, tok["attention_mask"])
            embs.append(pooled.detach().cpu())

        return torch.cat(embs, dim=0)
    # def _embed_texts(self, texts: List[str]) -> torch.Tensor:
    #     embs: List[torch.Tensor] = []
    #     bs = self.cfg.batch_size
    #     for i in range(0, len(texts), bs):
    #         batch = texts[i:i + bs]
    #         tok = self.tokenizer(
    #             batch,
    #             padding=True,
    #             truncation=True,
    #             max_length=self.cfg.max_length,
    #             return_tensors="pt",
    #         ).to(self.cfg.device)

    #         out = self.model(**tok, return_dict=True)
    #         pooled = self._mean_pool(out.last_hidden_state, tok["attention_mask"])  # (B, H)
    #         embs.append(pooled.detach().cpu())
    #     return torch.cat(embs, dim=0)

    def topk_candidates(self, s: str, r_raw: str, o: str) -> List[Candidate]:
        q = f"rel: {r_raw} | subj: {s} | obj: {o}"
        q = self.cfg.query_prefix + q

        with torch.inference_mode():
            q_emb = self._embed_texts([q])  # (1, D) on CPU
            if self.cfg.normalize:
                q_emb = F.normalize(q_emb, p=2, dim=1)

            scores = (self.schema_emb @ q_emb[0]).float()  # (N,)
            k = min(self.cfg.top_k, scores.numel())
            topv, topi = torch.topk(scores, k=k, largest=True)

        cands: List[Candidate] = []
        for score, idx in zip(topv.tolist(), topi.tolist()):
            it = self.schema_items[idx]
            cands.append(Candidate(name=it.name, definition=it.definition, score=float(score)))
        return cands