# oie/framework.py
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import pathlib
import os
import torch
import math

from tqdm import tqdm

from .oie import OIEConfig, OIEEngine
from .complementation import ComplementationConfig, ComplementationEngine
from .utils.rgat_complementation import elbow_cut_top_pairs, build_complementation_hints, triples_to_set
from .utils.schema_retriever import SchemaRetriever, SchemaRetrieverConfig
from .utils.schema_reranker import SchemaReranker, SchemaRerankerConfig

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    rgat_pairs_jsonl: str

    # Step1: OIE
    oie_llm: str
    oie_prompt_template_file_path: str
    oie_few_shot_example_file_path: str

    out_jsonl: str = "oie_outputs.jsonl"
    dump_pretty_json: bool = False
    pretty_json_path: str | None = None
    loglevel: int = logging.INFO

    # Step2: Complementation (optional)
    run_step2: bool = False
    comp_llm: str = "meta-llama/Llama-3.1-8B-Instruct"
    comp_prompt_template_file_path: str = "./prompt_templates/oie_complementation.txt"
    comp_few_shot_example_file_path: str = "./few_shot_examples/example/complementation_few_shot_examples.txt"
    elbow_min_k: int = 2
    elbow_max_k: int = 8

    # Step3: Schema normalization (bi-encoder top-k + cross-encoder rerank) (optional)
    # NOTE: 기존 필드명 run_closedie를 그대로 쓰지만, 현재 구현은 LLM 없이 rerank-only임.
    run_closedie: bool = False
    schema_csv_path: str | None = None

    # Retriever (bi-encoder)
    schema_embed_model: str = "intfloat/multilingual-e5-base"
    schema_top_k: int = 10  # default=10으로 변경
    schema_batch_size: int = 64
    schema_max_length: int = 128

    # Reranker (cross-encoder)
    schema_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    schema_reranker_batch_size: int = 32
    schema_reranker_max_length: int = 256

    # 여기 잘 봐야됨
    ce_abs_threshold: float = -1.0
    ce_margin_threshold: float = 0.0
    ce_gate_mode: str = "abs"
    ce_abs_quantile: float | None = None  # e.g., 0.55
    fallback_band_quantile: float | None = None  # e.g., 0.50
    fallback_margin_quantile: float | None = None  # e.g., 0.40
    quantile_clip_low: float | None = None       # e.g., -8.0
    quantile_clip_high: float | None = None      # e.g., 0.0

    # 정책: 정규화가 실패(후보 없음 등)할 때 어떻게 할지
    # - "drop": 해당 트리플 제거
    # - "keep_raw": 원래 R_raw 유지
    closedie_none_policy: str = "drop"  # or "keep_raw"
    closedie_log_all_meta: bool = False

    # fallback(salvage) params
    fallback_base_abs: float | None = None
    fallback_band_low: float = -3.5
    fallback_margin_min: float = 0.8
    fallback_max_add: int = 1
    fallback_require_evidence: bool = False


class OpenIEFramework:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        logger.setLevel(cfg.loglevel)

        # Step1 engine
        self.oie_engine = OIEEngine(
            OIEConfig(
                oie_llm=cfg.oie_llm,
                oie_prompt_template_file_path=cfg.oie_prompt_template_file_path,
                oie_few_shot_example_file_path=cfg.oie_few_shot_example_file_path,
                loglevel=cfg.loglevel,
            )
        )

        # Step2 engine placeholder
        self.comp_engine: Optional[ComplementationEngine] = None

        # Step3 resources placeholders (lazy)
        self.schema_retriever: Optional[SchemaRetriever] = None
        self.schema_reranker: Optional[SchemaReranker] = None

    @staticmethod
    def _validate_sample(obj: Dict[str, Any]) -> None:
        if "input" not in obj:
            raise KeyError('Missing required key "input"')
        if "atomic" not in obj:
            raise KeyError('Missing required key "atomic"')

        if not isinstance(obj["input"], str):
            raise TypeError(f'"input" must be str, got {type(obj["input"])}')

        atomic = obj["atomic"]
        if not isinstance(atomic, list) or not all(isinstance(x, str) for x in atomic):
            raise TypeError('"atomic" must be List[str]')

        if "top_pairs" in obj and not isinstance(obj["top_pairs"], list):
            raise TypeError('"top_pairs" must be List[...] if provided')

    def _dump_pretty_json(self) -> None:
        pretty_path = (
            self.cfg.pretty_json_path
            if self.cfg.pretty_json_path is not None
            else str(pathlib.Path(self.cfg.out_jsonl).with_suffix(".json"))
        )

        records = []
        with open(self.cfg.out_jsonl, "r", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    records.append(json.loads(line))

        with open(pretty_path, "w", encoding="utf-8") as fout:
            json.dump(records, fout, ensure_ascii=False, indent=2)

    # -------------------------
    # Step3 helpers (rerank-only)
    # -------------------------
    def _init_closedie_resources_if_needed(self):
        """
        NOTE: 함수명은 기존 코드 호환을 위해 유지.
        실제 동작은 (SchemaRetriever + SchemaReranker) 초기화임.
        """
        if not self.cfg.run_closedie:
            return

        if not self.cfg.schema_csv_path:
            raise ValueError("run_closedie=true but schema_csv_path is not set.")

        # init retriever
        if self.schema_retriever is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.schema_retriever = SchemaRetriever(
                SchemaRetrieverConfig(
                    schema_csv_path=self.cfg.schema_csv_path,
                    embed_model_name_or_path=self.cfg.schema_embed_model,
                    top_k=self.cfg.schema_top_k,
                    batch_size=self.cfg.schema_batch_size,
                    max_length=self.cfg.schema_max_length,
                    device=device,
                    normalize=True,
                )
            )

        # init reranker
        if self.schema_reranker is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.schema_reranker = SchemaReranker(
                SchemaRerankerConfig(
                    reranker_model_name_or_path=self.cfg.schema_reranker_model,
                    batch_size=self.cfg.schema_reranker_batch_size,
                    max_length=self.cfg.schema_reranker_max_length,
                    device=device,
                )
            )
    def _compute_quantile(self, values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        q = max(0.0, min(1.0, float(q)))
        vs = sorted(float(x) for x in values)
        n = len(vs)
        if n == 1:
            v = vs[0]
        else:
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                v = vs[lo]
            else:
                w = pos - lo
                v = (1.0 - w) * vs[lo] + w * vs[hi]

        # optional clipping
        if self.cfg.quantile_clip_low is not None:
            v = max(v, float(self.cfg.quantile_clip_low))
        if self.cfg.quantile_clip_high is not None:
            v = min(v, float(self.cfg.quantile_clip_high))
        return float(v)

    def _score_triples_closedie(self, triples: List[List[str]]) -> Tuple[List[float], List[float]]:
        """
        rerank 점수 분포(best_ce, margin)만 모으기 위한 scoring.
        반환: best_ce_list, margin_list(m>1일 때만)
        """
        assert self.schema_retriever is not None
        assert self.schema_reranker is not None

        best_list: List[float] = []
        margin_list: List[float] = []

        for t in triples or []:
            if not (isinstance(t, list) and len(t) == 3):
                continue
            s, r_raw, o = (t[0] or "").strip(), (t[1] or "").strip(), (t[2] or "").strip()
            if not s or not r_raw or not o:
                continue

            cands = self.schema_retriever.topk_candidates(s, r_raw, o)
            if not cands:
                continue

            query_text = f"rel: {r_raw} | subj: {s} | obj: {o}"
            cand_texts = [f"{c.name}: {c.definition}" for c in cands]
            ce_scores = self.schema_reranker.score(query_text, cand_texts)

            m = min(len(cands), len(ce_scores))
            if m <= 0:
                continue

            ce_scores = ce_scores[:m]
            sorted_idx = sorted(range(m), key=lambda i: ce_scores[i], reverse=True)

            best_ce = float(ce_scores[sorted_idx[0]])
            best_list.append(best_ce)

            if m > 1:
                second_ce = float(ce_scores[sorted_idx[1]])
                margin_list.append(best_ce - second_ce)

        return best_list, margin_list

    def _calibrate_quantile_thresholds(self, step_path: pathlib.Path) -> None:
        """
        qabs/qabs+margin 모드에서:
        - ce_abs_threshold      <- quantile(best_ce, ce_abs_quantile)
        - fallback_band_low     <- quantile(best_ce, fallback_band_quantile)
        - fallback_margin_min   <- quantile(margin,  fallback_margin_quantile)
        """
        gate_mode = (self.cfg.ce_gate_mode or "abs").lower()
        if not gate_mode.startswith("q"):
            return

        # quantile 값이 하나도 없으면 할 게 없음
        if self.cfg.ce_abs_quantile is None and self.cfg.fallback_band_quantile is None and self.cfg.fallback_margin_quantile is None:
            return

        best_all: List[float] = []
        margin_all: List[float] = []

        n_rec = 0
        with open(step_path, "r", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                n_rec += 1

                rec = json.loads(line)
                oie_triples = rec.get("oie_triples")
                comp_triples = rec.get("complemented_triples")

                if isinstance(oie_triples, list) and oie_triples:
                    b, m = self._score_triples_closedie(oie_triples)
                    best_all.extend(b)
                    margin_all.extend(m)

                if isinstance(comp_triples, list) and comp_triples:
                    b, m = self._score_triples_closedie(comp_triples)
                    best_all.extend(b)
                    margin_all.extend(m)

        abs_thr = None
        band_low = None
        margin_min = None

        if self.cfg.ce_abs_quantile is not None:
            abs_thr = self._compute_quantile(best_all, float(self.cfg.ce_abs_quantile))

        if self.cfg.fallback_band_quantile is not None:
            band_low = self._compute_quantile(best_all, float(self.cfg.fallback_band_quantile))

        if self.cfg.fallback_margin_quantile is not None:
            if len(margin_all) > 0:
                margin_min = self._compute_quantile(margin_all, float(self.cfg.fallback_margin_quantile))
            else:
                logger.warning("[qabs] margin calibration skipped: no margin points (all candidate lists had size 1).")

        # sanity: band_low는 abs_thr보다 낮아야 의미가 있음
        if abs_thr is not None and band_low is not None and band_low >= abs_thr:
            band_low = abs_thr - 1e-6

        if abs_thr is not None:
            self.cfg.ce_abs_threshold = float(abs_thr)
        if band_low is not None:
            self.cfg.fallback_band_low = float(band_low)
        if margin_min is not None:
            self.cfg.fallback_margin_min = float(margin_min)

        logger.info(
            f"[qabs] calibrated: ce_abs_threshold={self.cfg.ce_abs_threshold:.4f}, "
            f"fallback_band_low={self.cfg.fallback_band_low:.4f}, "
            f"fallback_margin_min={self.cfg.fallback_margin_min:.4f} "
            f"(best_points={len(best_all)}, margin_points={len(margin_all)}, records={n_rec})"
        )

    def _normalize_triples_with_closedie(
        self,
        triples: List[List[str]],
    ) -> Tuple[List[List[str]], List[Dict[str, Any]]]:
        """
        NOTE: 함수명은 기존 코드 호환을 위해 유지.
        실제 동작은:
          - bi-encoder로 top-k 후보 생성
          - cross-encoder reranker로 top-1 선택
          - meta에 top-k/점수/마진 로그 저장
        """
        assert self.schema_retriever is not None
        assert self.schema_reranker is not None

        normalized: List[List[str]] = []
        metas: List[Dict[str, Any]] = []

        for t in triples:
            if not (isinstance(t, list) and len(t) == 3):
                metas.append({"skipped": True, "reason": "invalid_triple"})
                continue

            s, r_raw, o = (t[0] or "").strip(), (t[1] or "").strip(), (t[2] or "").strip()

            # L1-A query text (로그용)
            query_text = f"rel: {r_raw} | subj: {s} | obj: {o}"

            # 1) bi-encoder top-k
            cands = self.schema_retriever.topk_candidates(s, r_raw, o)
            if not cands:
                # 후보 자체가 없으면 정책 적용
                if self.cfg.closedie_none_policy == "drop":
                    metas.append({
                        "triple": [s, r_raw, o],
                        "query_text": query_text,
                        "candidates": [],
                        "choice": None,
                        "dropped": True,
                        "reason": "no_candidates",
                    })
                    continue
                else:
                    normalized.append([s, r_raw, o])
                    metas.append({
                        "triple": [s, r_raw, o],
                        "query_text": query_text,
                        "candidates": [],
                        "choice": "KEEP_RAW",
                        "dropped": False,
                        "reason": "no_candidates_keep_raw",
                    })
                    continue

            # reranker 입력 candidate text (schema retriever의 it.text와 동일 포맷이 가장 깔끔)
            cand_texts = [f"{c.name}: {c.definition}" for c in cands]

            # 2) cross-encoder rerank
            ce_scores = self.schema_reranker.score(query_text, cand_texts)

            # 안전장치: 길이 불일치시 min 길이로 자름
            m = min(len(cands), len(ce_scores))
            cands = cands[:m]
            ce_scores = ce_scores[:m]

            if m == 0:
                if self.cfg.closedie_none_policy == "drop":
                    metas.append({
                        "triple": [s, r_raw, o],
                        "query_text": query_text,
                        "candidates": [],
                        "choice": None,
                        "dropped": True,
                        "reason": "reranker_no_scores",
                    })
                    continue
                else:
                    normalized.append([s, r_raw, o])
                    metas.append({
                        "triple": [s, r_raw, o],
                        "query_text": query_text,
                        "candidates": [],
                        "choice": "KEEP_RAW",
                        "dropped": False,
                        "reason": "reranker_no_scores_keep_raw",
                    })
                    continue

            # 3) select best / margin
            # sorted_idx = sorted(range(m), key=lambda i: ce_scores[i], reverse=True)
            # best_i = sorted_idx[0]
            # best_ce = float(ce_scores[best_i])
            # second_ce = float(ce_scores[sorted_idx[1]]) if m > 1 else None
            # margin = (best_ce - second_ce) if second_ce is not None else None

            # choice = cands[best_i].name
            # normalized.append([s, choice, o])

            # metas.append({
            #     "triple": [s, r_raw, o],
            #     "query_text": query_text,
            #     "top_k": m,
            #     "choice": choice,
            #     "best_ce_score": best_ce,
            #     "second_ce_score": second_ce,
            #     "ce_margin": margin,
            #     "dropped": False,
            #     "candidates": [
            #         {
            #             "name": c.name,
            #             "definition": c.definition,
            #             "bi_score": float(c.score),
            #             "ce_score": float(ce_scores[i]),
            #         }
            #         for i, c in enumerate(cands)
            #     ],
            # })
            sorted_idx = sorted(range(m), key=lambda i: ce_scores[i], reverse=True)
            best_i = sorted_idx[0]
            best_ce = float(ce_scores[best_i])
            second_ce = float(ce_scores[sorted_idx[1]]) if m > 1 else None
            margin = (best_ce - second_ce) if second_ce is not None else None
            best_choice = cands[best_i].name
            best_choice_definition = cands[best_i].definition

            # --- NEW: NONE gate (A) ---
            gate_reasons = []
            gate_mode = (self.cfg.ce_gate_mode or "abs").lower()
            if gate_mode in ("abs", "abs+margin", "qabs", "qabs+margin"):
                if self.cfg.ce_abs_threshold is not None and best_ce < float(self.cfg.ce_abs_threshold):
                    gate_reasons.append(f"abs<{self.cfg.ce_abs_threshold}")

            if gate_mode in ("margin", "abs+margin", "qabs+margin"):
                if margin is not None and self.cfg.ce_margin_threshold is not None:
                    if margin < float(self.cfg.ce_margin_threshold):
                        gate_reasons.append(f"margin<{self.cfg.ce_margin_threshold}")

            if gate_reasons:
                # NONE 판정 -> 정책 적용
                if self.cfg.closedie_none_policy == "drop":
                    metas.append({
                        "triple": [s, r_raw, o],
                        "query_text": query_text,
                        "top_k": m,
                        "choice": None,
                        "best_choice": best_choice,
                        "best_choice_definition": best_choice_definition,
                        "dropped": True,
                        "reason": "none_gate",
                        "gate_mode": self.cfg.ce_gate_mode,
                        "gate_reasons": gate_reasons,
                        "best_ce_score": best_ce,
                        "second_ce_score": second_ce,
                        "ce_margin": margin,
                        "candidates": [
                            {
                                "name": c.name,
                                "definition": c.definition,
                                "bi_score": float(c.score),
                                "ce_score": float(ce_scores[i]),
                            }
                            for i, c in enumerate(cands)
                        ],
                    })
                    continue
                else:
                    # keep_raw
                    normalized.append([s, r_raw, o])
                    metas.append({
                        "triple": [s, r_raw, o],
                        "query_text": query_text,
                        "top_k": m,
                        "choice": "KEEP_RAW",
                        "dropped": False,
                        "reason": "none_gate_keep_raw",
                        "gate_mode": self.cfg.ce_gate_mode,
                        "gate_reasons": gate_reasons,
                        "best_ce_score": best_ce,
                        "second_ce_score": second_ce,
                        "ce_margin": margin,
                        "candidates": [
                            {
                                "name": c.name,
                                "definition": c.definition,
                                "bi_score": float(c.score),
                                "ce_score": float(ce_scores[i]),
                            }
                            for i, c in enumerate(cands)
                        ],
                    })
                    continue

            # --- gate 통과한 경우만 relation 선택 ---
            choice = cands[best_i].name
            normalized.append([s, choice, o])

            # (optional) success meta logging
            if self.cfg.closedie_log_all_meta:
                metas.append({
                    "triple": [s, r_raw, o],
                    "query_text": query_text,
                    "top_k": m,
                    "choice": choice,
                    "dropped": False,
                    "gate_mode": self.cfg.ce_gate_mode,
                    "best_ce_score": best_ce,
                    "second_ce_score": second_ce,
                    "ce_margin": margin,
                    "candidates": [
                        {
                            "name": c.name,
                            "definition": c.definition,
                            "bi_score": float(c.score),
                            "ce_score": float(ce_scores[i]),
                        }
                        for i, c in enumerate(cands)
                    ],
                })

        return normalized, metas
    
    def _evidence_hit(self, s: str, o: str, atomic: List[str]) -> bool:
        s = (s or "").strip()
        o = (o or "").strip()
        if not s or not o or not atomic:
            return False

        same = any((s in af and o in af) for af in atomic)
        if same:
            return True

        has_s = any(s in af for af in atomic)
        has_o = any(o in af for af in atomic)
        return has_s and has_o
    
    def _salvage_from_dropped(
        self,
        normalized: List[List[str]],
        metas: List[Dict[str, Any]],
        atomic: List[str],
        base_abs: float = -2.5,
        band_low: float = -3.5,
        margin_min: float = 0.8,
        max_add: int = 1,
        require_evidence: bool = True,
    ) -> Tuple[List[List[str]], List[Dict[str, Any]]]:

        seen = set()
        for t in normalized or []:
            if isinstance(t, list) and len(t) == 3:
                seen.add((t[0].strip(), t[1].strip(), t[2].strip()))

        pool = []
        for i, m in enumerate(metas or []):
            if not isinstance(m, dict):
                continue
            if not m.get("dropped", False):
                continue
            if m.get("reason") != "none_gate":
                continue

            triple = m.get("triple")
            best_ce = m.get("best_ce_score")
            margin = m.get("ce_margin")
            best_choice = m.get("best_choice")

            if not triple or best_ce is None or margin is None or not best_choice:
                continue

            s, r_raw, o = triple
            rel_l = best_choice.strip().lower()

            if not (float(band_low) <= float(best_ce) < float(base_abs)):
                continue
            if float(margin) < float(margin_min):
                continue

            if require_evidence and (not self._evidence_hit(s, o, atomic)):
                continue

            salvage_score = (float(best_ce) - float(band_low)) + 0.5 * float(margin)
            if self._evidence_hit(s, o, atomic):
                salvage_score += 1.0

            pool.append((salvage_score, i, s, best_choice, o))

        pool.sort(reverse=True, key=lambda x: x[0])

        added = 0
        for _, meta_idx, s, choice, o in pool:
            if added >= max_add:
                break

            key = (s.strip(), choice.strip(), o.strip())
            if key in seen:
                continue

            normalized.append([s, choice, o])
            seen.add(key)

            metas[meta_idx]["salvaged"] = True
            metas[meta_idx]["salvage_reason"] = "fallback_band"
            added += 1

        return normalized, metas
    
    def _cleanup_step4(self):
        # schema retriever
        if self.schema_retriever is not None:
            try:
                self.schema_retriever.close()
            except Exception:
                pass
            self.schema_retriever = None

        # schema reranker
        if self.schema_reranker is not None:
            try:
                self.schema_reranker.close()
            except Exception:
                pass
            self.schema_reranker = None

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    # -------------------------
    # main
    # -------------------------
    def run(self):
        # -------------------
        # Step1: OpenIE
        # -------------------
        logger.info("[Step1] OpenIE run...")

        out_path = pathlib.Path(self.cfg.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.cfg.rgat_pairs_jsonl, "r", encoding="utf-8") as fcnt:
            total = sum(1 for line in fcnt if line.strip())

        n_written = 0

        with open(self.cfg.rgat_pairs_jsonl, "r", encoding="utf-8") as fin, \
             open(self.cfg.out_jsonl, "w", encoding="utf-8") as fout:

            pbar = tqdm(
                total=total,
                desc="OpenIE",
                unit="samples",
                dynamic_ncols=True,
                mininterval=1.0,
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )

            for line in fin:
                if not line.strip():
                    continue

                obj = json.loads(line)
                self._validate_sample(obj)

                idx = obj.get("index", n_written)
                raw = obj["input"]
                atomic = obj["atomic"]
                top_pairs = obj.get("top_pairs", [])

                try:
                    triples = self.oie_engine.extract_one(raw, atomic)
                    record = {
                        "index": idx,
                        "input": raw,
                        "atomic": atomic,
                        "top_pairs": top_pairs,
                        "oie_triples": triples,
                        "atomic_facts_str": "\n".join([f"{i}. {af.strip()}" for i, af in enumerate(atomic, start=1)]),
                    }
                except Exception as e:
                    record = {
                        "index": idx,
                        "input": raw,
                        "atomic": atomic,
                        "top_pairs": top_pairs,
                        "oie_triples": None,
                        "error": repr(e),
                    }

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1
                pbar.update(1)

            pbar.close()

        logger.info("[Step1] OIE Done.")

        self.oie_engine.close()
        try:
            del self.oie_engine
        except Exception:
            pass

        # -------------------
        # Step2: Complementation (optional)
        # -------------------
        if not self.cfg.run_step2:
            logger.info("[Step2] Skipped (run_step2=false).")
        else:
            self.comp_engine = ComplementationEngine(
                ComplementationConfig(
                    llm_name_or_path=self.cfg.comp_llm,
                    prompt_template_file_path=self.cfg.comp_prompt_template_file_path,
                    few_shot_example_file_path=self.cfg.comp_few_shot_example_file_path,
                    loglevel=self.cfg.loglevel,
                )
            )

            logger.info("[Step2] RGAT-based complementation run")

            step_path = pathlib.Path(self.cfg.out_jsonl)
            if not step_path.exists():
                raise FileNotFoundError(f"Step1 output not found: {step_path}")

            tmp_path = step_path.with_suffix(step_path.suffix + ".step2.tmp")

            with open(step_path, "r", encoding="utf-8") as fcnt:
                total2 = sum(1 for line in fcnt if line.strip())

            with open(step_path, "r", encoding="utf-8") as fin, \
                 open(tmp_path, "w", encoding="utf-8") as fout:

                pbar2 = tqdm(
                    total=total2,
                    desc="Complementation",
                    unit="samples",
                    dynamic_ncols=True,
                    mininterval=1.0,
                    leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )

                for line in fin:
                    if not line.strip():
                        continue

                    rec: Dict[str, Any] = json.loads(line)

                    raw = rec.get("input", "")
                    top_pairs = rec.get("top_pairs", [])
                    oie_triples = rec.get("oie_triples", None)

                    rec.setdefault("complemented_triples", [])
                    rec.setdefault("step2_meta", None)

                    if not isinstance(oie_triples, list):
                        rec["complemented_triples"] = None
                        rec["step2_meta"] = {"skipped": True, "reason": "step1_failed_or_no_triples"}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        pbar2.update(1)
                        continue

                    if not isinstance(top_pairs, list) or len(top_pairs) == 0:
                        rec["complemented_triples"] = []
                        rec["step2_meta"] = {"skipped": True, "reason": "no_top_pairs"}
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        pbar2.update(1)
                        continue

                    try:
                        elbow = elbow_cut_top_pairs(
                            top_pairs=top_pairs,
                            min_k=self.cfg.elbow_min_k,
                            max_k=self.cfg.elbow_max_k,
                        )
                        hints = build_complementation_hints(
                            elbow.selected_pairs,
                            include_relation_defs=True,
                        )

                        new_triples = self.comp_engine.extract_new_triples(
                            raw_sentence=raw,
                            existing_triples=oie_triples,
                            complementation_hints=hints,
                            few_shot_examples=self.comp_engine.few_shot_examples,
                        )

                        existing_set = triples_to_set(oie_triples)
                        filtered = []
                        for t in new_triples or []:
                            if not (isinstance(t, list) and len(t) == 3):
                                continue
                            s, r, o = t[0].strip(), t[1].strip(), t[2].strip()
                            if (s, r, o) in existing_set:
                                continue
                            filtered.append([s, r, o])

                        rec["complemented_triples"] = filtered
                        rec["step2_meta"] = {
                            "elbow_k": elbow.k,
                            "elbow_reason": elbow.reason,
                            "pair_scores_desc": elbow.scores,
                        }

                    except Exception as e:
                        rec["complemented_triples"] = None
                        rec["step2_meta"] = {"error": repr(e)}

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pbar2.update(1)

                pbar2.close()

            os.replace(tmp_path, step_path)

            logger.info("[Step2] Complementation Done. Output updated in-place.")

            # Step2 모델 정리
            self.comp_engine.close()
            try:
                del self.comp_engine
            except Exception:
                pass

        # -------------------
        # Step3: Schema normalization (optional) - rerank-only
        # -------------------
        if not self.cfg.run_closedie:
            logger.info("[Step3] Skipped (run_closedie=false).")
        else:
            logger.info("[Step3] Triples normalize run (bi-topk + ce-rerank)")

            self._init_closedie_resources_if_needed()

            step_path = pathlib.Path(self.cfg.out_jsonl)
            if not step_path.exists():
                raise FileNotFoundError(f"Output not found: {step_path}")
            self._calibrate_quantile_thresholds(step_path)
            tmp_path4 = step_path.with_suffix(step_path.suffix + ".step4.tmp")

            # (1) total3 계산 (Step1/2처럼)
            with open(step_path, "r", encoding="utf-8") as fcnt:
                total3 = sum(1 for line in fcnt if line.strip())

            with open(step_path, "r", encoding="utf-8") as fin, open(tmp_path4, "w", encoding="utf-8") as fout:
                pbar3 = tqdm(
                    total=total3,
                    desc="Normalize(Step3)",
                    unit="samples",
                    dynamic_ncols=True,
                    mininterval=1.0,
                    leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                )

                for line in fin:
                    if not line.strip():
                        continue

                    rec: Dict[str, Any] = json.loads(line)

                    oie_triples = rec.get("oie_triples", None)
                    comp_triples = rec.get("complemented_triples", None)

                    # (선택) 이번 샘플에서 처리할 triple 개수 찍기
                    n_t1 = len(oie_triples) if isinstance(oie_triples, list) else 0
                    n_t2 = len(comp_triples) if isinstance(comp_triples, list) else 0
                    pbar3.set_postfix({"t1": n_t1, "t2": n_t2})

                    # normalize step1 triples
                    if isinstance(oie_triples, list):
                        norm1, meta1 = self._normalize_triples_with_closedie(oie_triples)

                        base_abs = self.cfg.fallback_base_abs
                        if base_abs is None:
                            base_abs = self.cfg.ce_abs_threshold

                        norm1, meta1 = self._salvage_from_dropped(
                            normalized=norm1,
                            metas=meta1,
                            atomic=rec.get("atomic", []),
                            base_abs=float(base_abs),
                            band_low=float(self.cfg.fallback_band_low),
                            margin_min=float(self.cfg.fallback_margin_min),
                            max_add=int(self.cfg.fallback_max_add),
                            require_evidence=bool(self.cfg.fallback_require_evidence),                            
                        )

                        rec["oie_triples_norm"] = norm1
                        rec["oie_triples_norm_meta"] = meta1
                    else:
                        rec["oie_triples_norm"] = None
                        rec["oie_triples_norm_meta"] = {"skipped": True, "reason": "no_step1_triples"}

                    # normalize step2 triples
                    if isinstance(comp_triples, list):
                        norm2, meta2 = self._normalize_triples_with_closedie(comp_triples)
                        rec["complemented_triples_norm"] = norm2
                        rec["complemented_triples_norm_meta"] = meta2

                        final: List[List[str]] = []
                        seen = set()

                        def _push(triples_):
                            if not isinstance(triples_, list):
                                return
                            for tt in triples_:
                                if not (isinstance(tt, list) and len(tt) == 3):
                                    continue
                                ss, rr, oo = (tt[0] or "").strip(), (tt[1] or "").strip(), (tt[2] or "").strip()
                                if not ss or not rr or not oo:
                                    continue
                                key = (ss, rr, oo)
                                if key in seen:
                                    continue
                                seen.add(key)
                                final.append([ss, rr, oo])

                        _push(rec.get("oie_triples_norm"))
                        _push(rec.get("complemented_triples_norm"))
                        rec["final_triples_norm"] = final
                    else:
                        rec["complemented_triples_norm"] = None
                        rec["complemented_triples_norm_meta"] = {
                            "skipped": True,
                            "reason": "no_step2_triples_or_step2_skipped",
                        }

                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    pbar3.update(1)

                pbar3.close()

            os.replace(tmp_path4, step_path)
            logger.info("[Step3] Triples normalize done. Output updated in-place.")

            self._cleanup_step4()

        # -------------------
        # Pretty dump at the end (single place)
        # -------------------
        if self.cfg.dump_pretty_json:
            self._dump_pretty_json()