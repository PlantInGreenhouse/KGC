import json
import os
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import softmax

logger = logging.getLogger(__name__)

REL2ID = {
    "ContentScope": 0,
    "MethodEnablement": 1,
    "Provenance": 2,
    "ContextLocale": 3,
    "TemporalGate": 4,
    "Rationale": 5,
}
NUM_BASE_RELS = len(REL2ID)
SELF_ID = NUM_BASE_RELS
MASK_ID = NUM_BASE_RELS + 1
NUM_RELS_WITH_MASK = NUM_BASE_RELS + 2


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def dump_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def id2rel(rel_id: int) -> str:
    for k, v in REL2ID.items():
        if v == rel_id:
            return k
    if rel_id == SELF_ID:
        return "SELF"
    if rel_id == MASK_ID:
        return "MASK"
    return f"REL_{rel_id}"


def build_graph_from_record(
    record: Dict[str, Any],
    node_emb: torch.Tensor,  # [N, D]
    add_reverse: bool = True,
    add_self_loops: bool = True,
) -> Data:
    atomic = record["atomic"]
    n = len(atomic)

    # base edges
    edges: List[Tuple[int, int, int, float]] = []
    for e in record.get("relations", []):
        rel = e.get("relation", "None")
        if rel not in REL2ID:
            continue
        s = int(e["af1_id"])
        t = int(e["af2_id"])
        if s < 0 or t < 0 or s >= n or t >= n:
            continue
        if s == t:
            continue
        conf = float(e.get("confidence", 1.0))
        rid = REL2ID[rel]
        edges.append((s, t, rid, conf))
        if add_reverse:
            edges.append((t, s, rid, conf))

    # self-loops (encoder stability / "자기 정보" 반영)
    if add_self_loops:
        for i in range(n):
            edges.append((i, i, SELF_ID, 1.0))

    if len(edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_type = torch.empty((0,), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
        y_edge = torch.empty((0,), dtype=torch.long)
        w_edge = torch.empty((0,), dtype=torch.float)
        is_base_edge = torch.empty((0,), dtype=torch.bool)
    else:
        src = torch.tensor([x[0] for x in edges], dtype=torch.long)
        dst = torch.tensor([x[1] for x in edges], dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)  # [2,E]
        edge_type = torch.tensor([x[2] for x in edges], dtype=torch.long)  # [E] 0..5 or SELF_ID
        conf = torch.tensor([x[3] for x in edges], dtype=torch.float)      # [E]
        edge_attr = conf.unsqueeze(-1)  # [E,1]

        # 학습 target은 base edge(0..5)만. self-loop는 예측 대상에서 제외.
        is_base_edge = edge_type < NUM_BASE_RELS
        y_edge = edge_type.clamp(min=0, max=NUM_BASE_RELS - 1).clone()  # placeholder
        w_edge = conf.clone()

    data = Data(
        x=node_emb,
        edge_index=edge_index,
        edge_type=edge_type,        # 0..5, SELF_ID(6)
        edge_attr=edge_attr,        # [E,1] confidence
        y_edge=y_edge,              # [E] (base edges만 의미 있음)
        w_edge=w_edge,              # [E] confidence weight
        is_base_edge=is_base_edge,  # [E] base edge mask
        num_nodes=n,
    )
    data.record_index = int(record.get("index", -1))
    data.raw_text = record.get("input", "")
    data.atomic = atomic
    return data


class TinyRGATLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        num_rels_with_mask: int,
        rel_emb_dim: int = 16,
        dropout: float = 0.3,
        conf_log_weight: float = 0.5,  # attention score에 log(conf) 반영 강도
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.num_rels = num_rels_with_mask
        self.conf_log_weight = conf_log_weight

        # shared linear transform (이미 encoder에서 hid로 들어오므로 여기서는 identity에 가까워도 됨)
        self.msg_lin = nn.Linear(hid_dim, hid_dim, bias=False)

        # relation embeddings (BASE + SELF + MASK)
        self.rel_emb = nn.Embedding(num_rels_with_mask, rel_emb_dim)
        self.rel_proj = nn.Linear(rel_emb_dim, hid_dim, bias=False)

        # attention vector (shared) over [h_dst || h_src || rel_vec]
        self.att_vec = nn.Parameter(torch.empty(3 * hid_dim))
        nn.init.xavier_uniform_(self.att_vec.unsqueeze(-1))

        self.leaky = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(hid_dim))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_in: torch.Tensor,
        edge_attr: torch.Tensor,
        return_alpha: bool = False,
    ):
        if edge_index.numel() == 0:
            out = x.new_zeros((x.size(0), self.hid_dim)) + self.bias
            if return_alpha:
                return out, x.new_zeros((0,))
            return out

        src, dst = edge_index[0], edge_index[1]
        t = edge_type_in.clamp(min=0, max=self.num_rels - 1)

        # node transforms
        h_src = self.msg_lin(x[src])
        h_dst = self.msg_lin(x[dst])

        # relation vec
        r = self.rel_proj(self.rel_emb(t))  # [E, hid]

        z = torch.cat([h_dst, h_src, r], dim=-1)  # [E, 3*hid]
        e = (z * self.att_vec).sum(dim=-1)
        e = self.leaky(e)

        # confidence를 attention에 "약하게" 반영: e += w * log(conf + eps)
        if self.conf_log_weight != 0.0:
            conf = edge_attr.squeeze(-1).clamp_min(1e-6)
            e = e + self.conf_log_weight * torch.log(conf)

        e = self.dropout(e)

        # dst 기준 정규화: "dst로 들어오는 이웃 중 누가 중요한가"
        alpha = softmax(e, dst)
        alpha = self.dropout(alpha)

        msg = h_src * alpha.unsqueeze(-1)
        out = x.new_zeros((x.size(0), self.hid_dim))
        out.index_add_(0, dst, msg)
        out = out + self.bias

        if return_alpha:
            return out, alpha
        return out


class TinyRGATEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int = 32,
        num_layers: int = 1,
        rel_emb_dim: int = 16,
        dropout: float = 0.3,
        conf_log_weight: float = 0.5,
    ):
        super().__init__()
        self.in_lin = nn.Linear(in_dim, hid_dim)
        self.layers = nn.ModuleList([
            TinyRGATLayer(
                hid_dim=hid_dim,
                num_rels_with_mask=NUM_RELS_WITH_MASK,
                rel_emb_dim=rel_emb_dim,
                dropout=dropout,
                conf_log_weight=conf_log_weight,
            )
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        data: Data,
        edge_type_in: torch.Tensor,
        return_alphas: bool = False,
    ):
        x = self.in_lin(data.x)
        x = F.relu(x)
        x = self.dropout(x)

        alphas: List[torch.Tensor] = []
        for layer, ln in zip(self.layers, self.norms):
            if return_alphas:
                h, a = layer(x, data.edge_index, edge_type_in, data.edge_attr, return_alpha=True)
                alphas.append(a)
            else:
                h = layer(x, data.edge_index, edge_type_in, data.edge_attr, return_alpha=False)
            x = ln(x + F.relu(h))
            x = self.dropout(x)

        if return_alphas:
            return x, alphas
        return x


class EdgeRelationHead(nn.Module):
    """
    masked edge-type prediction head (BASE 6개만 예측)
    self-loop edge는 학습 대상에서 제외(is_base_edge로 필터)
    """
    def __init__(self, hid_dim: int, num_rels: int = NUM_BASE_RELS, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hid_dim + 1, hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, num_rels),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return h.new_zeros((0, NUM_BASE_RELS))
        src, dst = edge_index[0], edge_index[1]
        x = torch.cat([h[src], h[dst], edge_attr], dim=-1)
        return self.mlp(x)


def weighted_ce_loss(logits: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if logits.size(0) == 0:
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, y, reduction="none")
    return (ce * w).mean()


@dataclass
class RGATConfig:
    e5_model: str = "intfloat/e5-base-v2"
    epochs: int = 300
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    hid_dim: int = 32
    num_layers: int = 1
    rel_emb_dim: int = 16
    dropout: float = 0.4
    mask_prob: float = 0.2
    conf_log_weight: float = 0.5
    use_conf_in_loss: bool = False
    add_reverse: bool = False
    no_self_loops: bool = False
    seed: int = 7
    patience: int = 40
    save_ckpt: str = "./rgat_ckpt.pt"
    top_k_pairs: int = 10
    loglevel: int = logging.INFO


def compute_in_deg(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.zeros((num_nodes,), dtype=torch.long)
    dst = edge_index[1]
    return torch.bincount(dst, minlength=num_nodes)


def score_pairs_from_alpha(
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_attr: torch.Tensor,
    alpha: torch.Tensor,
    atomic: List[str],
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    강한 연결 노드쌍 점수:
      - base edge(0..5)만 대상으로
      - s(i->j) = alpha(i->j) * conf
      - 무방향 pair는 양방향 평균(있으면)으로 집계
    """
    if edge_index.numel() == 0:
        return []

    src = edge_index[0].cpu().tolist()
    dst = edge_index[1].cpu().tolist()
    et = edge_type.cpu().tolist()
    conf = edge_attr.squeeze(-1).cpu().tolist()
    a = alpha.cpu().tolist()

    # 방향 점수 저장
    dir_score: Dict[Tuple[int, int], float] = {}
    dir_meta: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for s, t, r, c, al in zip(src, dst, et, conf, a):
        if r >= NUM_BASE_RELS:  # self-loop 등 제외
            continue
        sc = float(al) * float(c)
        key = (s, t)
        prev = dir_score.get(key, -1.0)

        if sc > prev:
            dir_score[key] = sc
            dir_meta[key] = {
                "src": s,
                "dst": t,
                "rel_id": r,
                "rel": [k for k, v in REL2ID.items() if v == r][0],
                "confidence": float(c),
                "alpha": float(al),
                "dir_score": float(sc),
                "src_text": atomic[s],
                "dst_text": atomic[t],
            }

    # 무방향 pair 집계
    pair_score: Dict[Tuple[int, int], float] = {}
    pair_detail: Dict[Tuple[int, int], Dict[str, Any]] = {}
    nodes = len(atomic)

    for i in range(nodes):
        for j in range(i + 1, nodes):
            s1 = dir_score.get((i, j), 0.0)
            s2 = dir_score.get((j, i), 0.0)
            if s1 == 0.0 and s2 == 0.0:
                continue
            und = 0.5 * (s1 + s2) if (s1 > 0 and s2 > 0) else max(s1, s2)
            pair_score[(i, j)] = und
            pair_detail[(i, j)] = {
                "pair": [i, j],
                "pair_score": float(und),
                "i_text": atomic[i],
                "j_text": atomic[j],
                "dir_ij": dir_meta.get((i, j)),
                "dir_ji": dir_meta.get((j, i)),
            }

    ranked = sorted(pair_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [pair_detail[k] for k, _ in ranked]


class RGATStrongPairsBuilder:
    def __init__(self, cfg: RGATConfig):
        self.cfg = cfg
        logger.setLevel(cfg.loglevel)

    def run(self, relations_jsonl: str, out_pairs_jsonl: str) -> None:
        cfg = self.cfg
        set_seed(cfg.seed)

        rows = load_jsonl(relations_jsonl)
        if not rows:
            raise ValueError("Empty relations jsonl.")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        embedder = SentenceTransformer(cfg.e5_model, device=device)

        graphs: List[Data] = []
        for r in tqdm(
            rows,
            total=len(rows),
            desc="RGAT: build graphs",
            dynamic_ncols=True,
            bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            atomic = r["atomic"]
            texts = ["passage: " + s for s in atomic]

            # 핵심: 내부 "Batches:" tqdm 끄기
            emb = embedder.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            g = build_graph_from_record(
                r,
                node_emb=emb,
                add_reverse=bool(cfg.add_reverse),
                add_self_loops=not bool(cfg.no_self_loops),
            )
            graphs.append(g)

        in_dim = graphs[0].x.size(1)

        encoder = TinyRGATEncoder(
            in_dim=in_dim,
            hid_dim=cfg.hid_dim,
            num_layers=cfg.num_layers,
            rel_emb_dim=cfg.rel_emb_dim,
            dropout=cfg.dropout,
            conf_log_weight=cfg.conf_log_weight,
        ).to(device)

        head = EdgeRelationHead(hid_dim=cfg.hid_dim, num_rels=NUM_BASE_RELS, dropout=cfg.dropout).to(device)

        optim = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loader = DataLoader(graphs, batch_size=cfg.batch_size, shuffle=True)

        best_loss = float("inf")
        bad_epochs = 0

        for epoch in range(1, cfg.epochs + 1):
            encoder.train()
            head.train()

            total_loss = 0.0
            total_masked = 0

            for batch in loader:
                batch = batch.to(device)
                E = batch.edge_type.size(0)
                if E == 0:
                    continue

                base_mask = batch.is_base_edge
                if base_mask.sum().item() == 0:
                    continue

                mask = (torch.rand(E, device=device) < cfg.mask_prob) & base_mask
                if mask.sum().item() == 0:
                    base_idx = torch.where(base_mask)[0]
                    mask[base_idx[torch.randint(0, base_idx.numel(), (1,), device=device)]] = True

                edge_type_in = batch.edge_type.clone()
                edge_type_in[mask] = MASK_ID

                h = encoder(batch, edge_type_in=edge_type_in, return_alphas=False)
                logits = head(h, batch.edge_index, batch.edge_attr)

                logits_m = logits[mask]
                y_m = batch.edge_type[mask]

                if cfg.use_conf_in_loss:
                    w_m = batch.edge_attr.squeeze(-1)[mask].clamp_min(1e-3)
                else:
                    w_m = torch.ones_like(y_m, dtype=torch.float, device=device)

                loss = weighted_ce_loss(logits_m, y_m, w_m)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), max_norm=1.0)
                optim.step()

                total_loss += float(loss.item()) * int(mask.sum().item())
                total_masked += int(mask.sum().item())

            avg = total_loss / max(1, total_masked)
            # logger.info(f"[RGAT][Epoch {epoch}/{cfg.epochs}] masked-edge loss={avg:.4f} (masked_edges={total_masked})")

            if avg + 1e-6 < best_loss:
                best_loss = avg
                bad_epochs = 0
                os.makedirs(os.path.dirname(cfg.save_ckpt) or ".", exist_ok=True)
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "head": head.state_dict(),
                        "cfg": cfg.__dict__,
                        "in_dim": in_dim,
                    },
                    cfg.save_ckpt,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    logger.info(f"[RGAT] Early stopping. Best loss={best_loss:.4f}")
                    break

        ckpt = torch.load(cfg.save_ckpt, map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        encoder.eval()

        pair_rows = []
        for g in tqdm(graphs, desc="RGAT: encode + score pairs"):
            g = g.to(device)
            with torch.no_grad():
                h, alphas = encoder(g, edge_type_in=g.edge_type, return_alphas=True)
                alpha_mean = torch.stack(alphas, dim=0).mean(dim=0) if len(alphas) > 1 else alphas[0]

            pairs = score_pairs_from_alpha(
                edge_index=g.edge_index.cpu(),
                edge_type=g.edge_type.cpu(),
                edge_attr=g.edge_attr.cpu(),
                alpha=alpha_mean.cpu(),
                atomic=g.atomic,
                top_k=cfg.top_k_pairs,
            )

            pair_rows.append({
                "index": int(g.record_index),
                "input": g.raw_text,
                "atomic": g.atomic,
                "top_pairs": pairs,
                "note": {
                    "reverse_edges": bool(cfg.add_reverse),
                    "self_loops": not bool(cfg.no_self_loops),
                    "conf_log_weight": float(cfg.conf_log_weight),
                    "use_conf_in_loss": bool(cfg.use_conf_in_loss),
                }
            })

        out_dir = os.path.dirname(out_pairs_jsonl) or "."
        os.makedirs(out_dir, exist_ok=True)
        dump_jsonl(out_pairs_jsonl, pair_rows)
        logger.info(f"[RGAT] Wrote strong pairs: {out_pairs_jsonl}")