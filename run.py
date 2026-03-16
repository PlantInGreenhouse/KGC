import argparse
import logging
import json
from pathlib import Path

from oie.framework import OpenIEFramework, RunConfig
from oie.preprocess.pipeline import PreprocessConfig, PreprocessPipeline


def build_parser():
    p = argparse.ArgumentParser()

    # -----------------------
    # NEW Step1~2: Preprocess (Atomic -> Weak links -> RGAT strong pairs)
    # -----------------------
    p.add_argument("--input_txt", type=str, default=None,
                   help="(NEW) Raw sentences txt. One sentence per line. If set and rgat_pairs_jsonl not given, preprocessing runs.")
    p.add_argument("--work_dir", type=str, default="./preprocess_outputs",
                   help="(NEW) Directory to store intermediate outputs (atomic/relations/rgat_pairs).")
    p.add_argument("--force_preprocess", action="store_true", default=False,
                   help="(NEW) Force regenerate preprocessing outputs even if files exist.")

    # Atomic decomposition
    p.add_argument("--atomic_llm", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="(NEW) LLM for atomic decomposition.")
    p.add_argument("--atomic_base_tokens", type=int, default=256)
    p.add_argument("--atomic_retries", type=int, default=2)

    # Weak link typing (LLM)
    p.add_argument("--weak_llm", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="(NEW) LLM for weak relation typing between atomic facts.")
    p.add_argument("--weak_batch_size", type=int, default=8)
    p.add_argument("--weak_max_new_tokens", type=int, default=256)
    p.add_argument("--weak_max_length", type=int, default=4096)
    p.add_argument("--weak_no_ficl", action="store_true", default=False,
                   help="(NEW) Disable few-shot turns for weak link typing.")

    # RGAT (strong pairs)
    p.add_argument("--rgat_e5_model", default="intfloat/e5-base-v2")
    p.add_argument("--rgat_epochs", type=int, default=300)
    p.add_argument("--rgat_batch_size", type=int, default=16)
    p.add_argument("--rgat_hid_dim", type=int, default=32)
    p.add_argument("--rgat_num_layers", type=int, default=1)
    p.add_argument("--rgat_rel_emb_dim", type=int, default=16)
    p.add_argument("--rgat_dropout", type=float, default=0.4)
    p.add_argument("--rgat_lr", type=float, default=5e-4)
    p.add_argument("--rgat_weight_decay", type=float, default=1e-4)
    p.add_argument("--rgat_mask_prob", type=float, default=0.2)
    p.add_argument("--rgat_conf_log_weight", type=float, default=0.0)
    p.add_argument("--rgat_use_conf_in_loss", action="store_true", default=False)
    p.add_argument("--rgat_add_reverse", action="store_true", default=False)
    p.add_argument("--rgat_no_self_loops", action="store_true", default=False)
    p.add_argument("--rgat_seed", type=int, default=7)
    p.add_argument("--rgat_patience", type=int, default=40)
    p.add_argument("--rgat_save_ckpt", type=str, default="./preprocess_outputs/rgat_ckpt.pt")
    p.add_argument("--rgat_top_k_pairs", type=int, default=10)
    p.add_argument("--rgat_rank_metric", type=str, default="e_center_sigmoid",
                   choices=["e_raw", "e_center", "e_center_sigmoid", "alpha", "alpha_in_deg"])

    # If user already has strong pairs, allow skipping preprocess
    p.add_argument("--rgat_pairs_jsonl", type=str, required=False, default=None,
                   help="(Optional) If set, skip preprocessing and use this file as input to existing OIE pipeline.")

    # -----------------------
    # Existing Step3~5 (old Step1~3): OpenIE settings
    # -----------------------
    p.add_argument(
        "--oie_llm",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM used for open information extraction.",
    )
    p.add_argument(
        "--oie_prompt_template",
        default="./prompt_templates/oie.txt",
        help="Prompt template for OIE (raw sentence + atomic facts).",
    )
    p.add_argument(
        "--oie_few_shot",
        default="./few_shot_examples/example/oie_few_shot_examples.txt",
        help="Few shot examples used for OpenIE.",
    )

    # -----------------------
    # Existing Step4 (old Step2): Complementation settings
    # -----------------------
    p.add_argument(
        "--run_step2",
        action="store_true",
        default=False,
        help="If set, run Step2 (RGAT-based complementation) after Step1, updating Step1 output JSONL in-place.",
    )
    p.add_argument(
        "--comp_llm",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="LLM used for complementation (Step2).",
    )
    p.add_argument(
        "--comp_prompt_template",
        default="./prompt_templates/oie_complementation.txt",
        help="Prompt template for Step2 complementation.",
    )
    p.add_argument(
        "--comp_few_shot",
        default="./few_shot_examples/example/complementation_few_shot_examples.txt",
        help="Few shot examples used for complementation.",
    )
    p.add_argument("--elbow_min_k", type=int, default=2)
    p.add_argument("--elbow_max_k", type=int, default=8)

    # -----------------------
    # Existing Step5 (old Step3): ClosedIE schema retrieval
    # -----------------------
    p.add_argument("--run_closedie", action="store_true", default=False,
                help="If set, run schema normalization (bi-encoder retrieval + cross-encoder rerank).")
    p.add_argument("--schema_csv", type=str, default=None)
    p.add_argument("--schema_embed_model", type=str, default="intfloat/multilingual-e5-base")
    p.add_argument("--schema_top_k", type=int, default=10)
    p.add_argument("--schema_batch_size", type=int, default=64)
    p.add_argument("--schema_max_length", type=int, default=4096)
    p.add_argument("--ce_abs_threshold", type=float, default=None)
    p.add_argument("--ce_margin_threshold", type=float, default=None)
    p.add_argument("--ce_gate_mode", type=str, default="abs", choices=["abs", "margin", "abs+margin", "qabs", "qabs+margin"])
    p.add_argument("--closedie_none_policy", type=str, default="drop", choices=["drop", "keep_raw"])
    p.add_argument("--closedie_log_all_meta", action="store_true", default=False)
    
    p.add_argument("--ce_abs_quantile", type=float, default=None,
               help="(qabs) Quantile for best_ce to set ce_abs_threshold. e.g., 0.55")
    p.add_argument("--fallback_band_quantile", type=float, default=None,
                help="Quantile for best_ce to set fallback_band_low. e.g., 0.50")
    p.add_argument("--fallback_margin_quantile", type=float, default=None,
                help="Quantile for margin (within band) to set fallback_margin_min. e.g., 0.40")
    p.add_argument("--quantile_clip_low", type=float, default=None)
    p.add_argument("--quantile_clip_high", type=float, default=None)
    
    p.add_argument("--fallback_base_abs", type=float, default=None)
    p.add_argument("--fallback_band_low", type=float, default=None)
    p.add_argument("--fallback_margin_min", type=float, default=None)
    p.add_argument("--fallback_max_add", type=int, default=1)
    p.add_argument("--fallback_require_evidence", action="store_true", default=False)
    p.add_argument("--schema_reranker_model", type=str, default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--schema_reranker_batch_size", type=int, default=32)
    p.add_argument("--schema_reranker_max_length", type=int, default=4096)

    # -----------------------
    # Outputs / misc
    # -----------------------
    p.add_argument("--out_jsonl", type=str, default="oie_outputs.jsonl")
    p.add_argument("--dump_pretty_json", action="store_true", default=False)
    p.add_argument("--pretty_json_path", type=str, default="./outputs/oie_outputs_pretty.json")
    p.add_argument("--loglevel", type=str, default="")
    p.add_argument("--kg_out_path", type=str, default=None)
    return p


def dump_final_kg(out_jsonl_path: str, kg_out_path: str) -> None:
    Path(kg_out_path).parent.mkdir(parents=True, exist_ok=True)

    n_lines = 0
    with open(out_jsonl_path, "r", encoding="utf-8") as fin, open(kg_out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            triples = rec.get("final_triples_norm", None)
            if not isinstance(triples, list):
                continue

            cleaned = []
            seen = set()
            for t in triples:
                if not (isinstance(t, list) and len(t) == 3):
                    continue
                s, r, o = (t[0] or "").strip(), (t[1] or "").strip(), (t[2] or "").strip()
                if not s or not r or not o:
                    continue
                key = (s, r, o)
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append([s, r, o])

            if not cleaned:
                continue

            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            n_lines += 1

    print(f"[KG] wrote {n_lines} lines -> {kg_out_path}")


def main():
    args = build_parser().parse_args()
    loglevel = getattr(logging, args.loglevel.upper(), logging.INFO)

    logging.basicConfig(level=loglevel, format="%(levelname)s: %(message)s", force=True)

    # -----------------------
    # NEW Step1~2: build rgat_pairs_jsonl if not provided
    # -----------------------
    rgat_pairs_path = args.rgat_pairs_jsonl
    if rgat_pairs_path is None:
        if not args.input_txt:
            raise ValueError("Either --rgat_pairs_jsonl or --input_txt must be provided.")
        pp_cfg = PreprocessConfig(
            input_txt=args.input_txt,
            work_dir=args.work_dir,
            force=args.force_preprocess,

            atomic_llm=args.atomic_llm,
            atomic_base_tokens=args.atomic_base_tokens,
            atomic_retries=args.atomic_retries,

            weak_llm=args.weak_llm,
            weak_batch_size=args.weak_batch_size,
            weak_max_new_tokens=args.weak_max_new_tokens,
            weak_max_length=args.weak_max_length,
            weak_no_ficl=args.weak_no_ficl,

            rgat_e5_model=args.rgat_e5_model,
            rgat_epochs=args.rgat_epochs,
            rgat_batch_size=args.rgat_batch_size,
            rgat_hid_dim=args.rgat_hid_dim,
            rgat_num_layers=args.rgat_num_layers,
            rgat_rel_emb_dim=args.rgat_rel_emb_dim,
            rgat_dropout=args.rgat_dropout,
            rgat_lr=args.rgat_lr,
            rgat_weight_decay=args.rgat_weight_decay,
            rgat_mask_prob=args.rgat_mask_prob,
            rgat_conf_log_weight=args.rgat_conf_log_weight,
            rgat_use_conf_in_loss=args.rgat_use_conf_in_loss,
            rgat_add_reverse=args.rgat_add_reverse,
            rgat_no_self_loops=args.rgat_no_self_loops,
            rgat_seed=args.rgat_seed,
            rgat_patience=args.rgat_patience,
            rgat_save_ckpt=args.rgat_save_ckpt,
            rgat_top_k_pairs=args.rgat_top_k_pairs,
            rgat_rank_metric=args.rgat_rank_metric,

            loglevel=loglevel,
        )
        pipeline = PreprocessPipeline(pp_cfg)
        rgat_pairs_path = pipeline.run()  # returns path to rgat_strong_pairs.jsonl

    # -----------------------
    # Existing Step3~5 (old Step1~3): unchanged OpenIEFramework usage
    # -----------------------
    cfg = RunConfig(
        rgat_pairs_jsonl=rgat_pairs_path,

        oie_llm=args.oie_llm,
        oie_prompt_template_file_path=args.oie_prompt_template,
        oie_few_shot_example_file_path=args.oie_few_shot,

        run_step2=args.run_step2,
        comp_llm=args.comp_llm,
        comp_prompt_template_file_path=args.comp_prompt_template,
        comp_few_shot_example_file_path=args.comp_few_shot,
        elbow_min_k=args.elbow_min_k,
        elbow_max_k=args.elbow_max_k,

        run_closedie=args.run_closedie,
        schema_csv_path=args.schema_csv,
        schema_embed_model=args.schema_embed_model,
        schema_top_k=args.schema_top_k,
        schema_batch_size=args.schema_batch_size,
        schema_max_length=args.schema_max_length,

        schema_reranker_model=args.schema_reranker_model,
        schema_reranker_batch_size=args.schema_reranker_batch_size,
        schema_reranker_max_length=args.schema_reranker_max_length,

        ce_abs_threshold=args.ce_abs_threshold,
        ce_margin_threshold=args.ce_margin_threshold,
        ce_gate_mode=args.ce_gate_mode,
        closedie_none_policy=args.closedie_none_policy,
        closedie_log_all_meta=args.closedie_log_all_meta,
        
        ce_abs_quantile=args.ce_abs_quantile,
        fallback_band_quantile=args.fallback_band_quantile,
        fallback_margin_quantile=args.fallback_margin_quantile,
        quantile_clip_low=args.quantile_clip_low,
        quantile_clip_high=args.quantile_clip_high,

        fallback_base_abs=args.fallback_base_abs,
        fallback_band_low=args.fallback_band_low,
        fallback_margin_min=args.fallback_margin_min,
        fallback_max_add=args.fallback_max_add,
        fallback_require_evidence=args.fallback_require_evidence,

        out_jsonl=args.out_jsonl,
        dump_pretty_json=args.dump_pretty_json,
        pretty_json_path=args.pretty_json_path,
        loglevel=loglevel,
    )

    fw = OpenIEFramework(cfg)
    fw.run()

    if args.kg_out_path:
        dump_final_kg(cfg.out_jsonl, args.kg_out_path)


if __name__ == "__main__":
    main()