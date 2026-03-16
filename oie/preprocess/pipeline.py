import logging
from dataclasses import dataclass
from pathlib import Path

from .atomic_decomp import AtomicDecompConfig, AtomicDecomposer
from .weak_links import WeakLinkConfig, WeakLinkTyper
from .rgat_strong_pairs import RGATConfig, RGATStrongPairsBuilder

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:    
    input_txt: str
    work_dir: str = "./preprocess_outputs"
    force: bool = False

    # Atomic
    atomic_llm: str = "meta-llama/Llama-3.1-8B-Instruct"
    atomic_base_tokens: int = 2048
    atomic_retries: int = 2

    # Weak links
    weak_llm: str = "meta-llama/Llama-3.1-8B-Instruct"
    weak_batch_size: int = 8
    weak_max_new_tokens: int = 4096
    weak_max_length: int = 8192
    weak_no_ficl: bool = False

    # RGAT
    rgat_e5_model: str = "intfloat/e5-base-v2"
    rgat_epochs: int = 300
    rgat_batch_size: int = 16
    rgat_hid_dim: int = 32
    rgat_num_layers: int = 1
    rgat_rel_emb_dim: int = 16
    rgat_dropout: float = 0.4
    rgat_lr: float = 5e-4
    rgat_weight_decay: float = 1e-4
    rgat_mask_prob: float = 0.2
    rgat_conf_log_weight: float = 0.0
    rgat_use_conf_in_loss: bool = False
    rgat_add_reverse: bool = True
    rgat_no_self_loops: bool = False
    rgat_seed: int = 7
    rgat_patience: int = 40
    rgat_save_ckpt: str = "./preprocess_outputs/rgat_ckpt.pt"
    rgat_top_k_pairs: int = 10
    rgat_rank_metric: str = "e_center_sigmoid"

    loglevel: int = logging.INFO


class PreprocessPipeline:
    """
    Produces rgat_strong_pairs.jsonl that matches OpenIEFramework's expected input:
      {"index","input","atomic","top_pairs", ...}
    """

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        logger.setLevel(cfg.loglevel)

        self.work_dir = Path(cfg.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.atomic_jsonl = self.work_dir / "atomic_output.jsonl"
        self.relations_jsonl = self.work_dir / "atomic_relations.jsonl"
        self.rgat_pairs_jsonl = self.work_dir / "rgat_strong_pairs.jsonl"

    def run(self) -> str:
        # Step1: Atomic decomposition
        if self.cfg.force or (not self.atomic_jsonl.exists()):
            logger.info(f"[Pre-Step1] Atomic decomposition -> {self.atomic_jsonl}")
            AtomicDecomposer(
                AtomicDecompConfig(
                    model_id=self.cfg.atomic_llm,
                    base_tokens=self.cfg.atomic_base_tokens,
                    retries=self.cfg.atomic_retries,
                    loglevel=self.cfg.loglevel,
                )
            ).run_txt(self.cfg.input_txt, str(self.atomic_jsonl))
        else:
            logger.info(f"[Pre-Step1] Skip (exists): {self.atomic_jsonl}")

        # Step2: Weak relation typing between atomic facts
        if self.cfg.force or (not self.relations_jsonl.exists()):
            logger.info(f"[Pre-Step2] Weak link typing -> {self.relations_jsonl}")
            WeakLinkTyper(
                WeakLinkConfig(
                    model_id=self.cfg.weak_llm,
                    batch_size=self.cfg.weak_batch_size,
                    max_new_tokens=self.cfg.weak_max_new_tokens,
                    max_length=self.cfg.weak_max_length,
                    no_ficl=self.cfg.weak_no_ficl,
                    loglevel=self.cfg.loglevel,
                )
            ).run_jsonl(str(self.atomic_jsonl), str(self.relations_jsonl))
        else:
            logger.info(f"[Pre-Step2] Skip (exists): {self.relations_jsonl}")

        # Step2-b: RGAT to get strong pairs
        if self.cfg.force or (not self.rgat_pairs_jsonl.exists()):
            logger.info(f"[Pre-Step2b] RGAT strong pairs -> {self.rgat_pairs_jsonl}")
            RGATStrongPairsBuilder(
                RGATConfig(
                    e5_model=self.cfg.rgat_e5_model,
                    epochs=self.cfg.rgat_epochs,
                    batch_size=self.cfg.rgat_batch_size,
                    hid_dim=self.cfg.rgat_hid_dim,
                    num_layers=self.cfg.rgat_num_layers,
                    rel_emb_dim=self.cfg.rgat_rel_emb_dim,
                    dropout=self.cfg.rgat_dropout,
                    lr=self.cfg.rgat_lr,
                    weight_decay=self.cfg.rgat_weight_decay,
                    mask_prob=self.cfg.rgat_mask_prob,
                    conf_log_weight=self.cfg.rgat_conf_log_weight,
                    use_conf_in_loss=self.cfg.rgat_use_conf_in_loss,
                    add_reverse=self.cfg.rgat_add_reverse,
                    no_self_loops=self.cfg.rgat_no_self_loops,
                    seed=self.cfg.rgat_seed,
                    patience=self.cfg.rgat_patience,
                    save_ckpt=self.cfg.rgat_save_ckpt,
                    top_k_pairs=self.cfg.rgat_top_k_pairs,
                    rank_metric=self.cfg.rgat_rank_metric,
                    loglevel=self.cfg.loglevel,
                )
            ).run(str(self.relations_jsonl), str(self.rgat_pairs_jsonl))
        else:
            logger.info(f"[Pre-Step2b] Skip (exists): {self.rgat_pairs_jsonl}")

        return str(self.rgat_pairs_jsonl)