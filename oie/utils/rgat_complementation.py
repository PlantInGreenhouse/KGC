from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional


# One-line relation definitions (use your provided semantics)
REL_DEFS_ONE_LINER: dict[str, str] = {
    # [What]
    "ContentScope": (
        "AF2 adjusts the content scope of AF1 by elaborating with clarifying details or summarizing into a higher-level gist."
    ),
    # [How]
    "MethodEnablement": (
        "AF2 provides the manner/means or enabling condition that explains how AF1 is achieved/performed/made possible."
    ),
    # [Who]
    "Provenance": (
        "AF2 explicitly attributes AF1 to a source/author/issuer, answering who said/wrote/created/reported it."
    ),
    # [Where]
    "ContextLocale": (
        "AF2 provides a where/inclusion background frame that situates where AF1 holds (within what larger entity)."
    ),
    # [When]
    "TemporalGate": (
        "AF2 provides a time anchor or temporal/conditional frame for when AF1 holds or happens."
    ),
    # [Why]
    "Rationale": (
        "AF2 provides an explicit cause/reason/explanation/evidence for AF1, or an explicit consequence/result of AF1."
    ),
}


@dataclass
class ElbowResult:
    k: int
    scores: List[float]
    selected_pairs: List[Dict[str, Any]]
    reason: str


def elbow_cut_top_pairs(
    top_pairs: List[Dict[str, Any]],
    min_k: int = 2,
    max_k: int = 8,
) -> ElbowResult:
    """
    Robust elbow cut for pair_score:
      1) sort by pair_score desc
      2) find largest drop between consecutive scores
      3) keep pairs up to the drop point (clamped by min_k/max_k)
    """
    pairs = [p for p in top_pairs if isinstance(p, dict) and "pair_score" in p]
    pairs.sort(key=lambda x: float(x["pair_score"]), reverse=True)
    scores = [float(p["pair_score"]) for p in pairs]

    n = len(scores)
    if n == 0:
        return ElbowResult(k=0, scores=[], selected_pairs=[], reason="no_pairs")

    if n <= min_k:
        return ElbowResult(k=n, scores=scores, selected_pairs=pairs[:n], reason="n<=min_k")

    deltas = [scores[i] - scores[i + 1] for i in range(n - 1)]
    j = max(range(len(deltas)), key=lambda i: deltas[i])
    k_elbow = j + 1

    k = max(min_k, min(k_elbow, max_k, n))

    # If all drops are tiny, fallback to max_k (but still <= n)
    if deltas[j] < 1e-3:
        k = min(max_k, n)
        return ElbowResult(k=k, scores=scores, selected_pairs=pairs[:k], reason="tiny_drop_fallback")

    return ElbowResult(k=k, scores=scores, selected_pairs=pairs[:k], reason=f"largest_drop@{j}_delta={deltas[j]:.6f}")


# -----------------------
# Hint building (core fix)
# -----------------------

def _safe_float(x: Any, default: float = float("-inf")) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _pick_best_direction(p: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    Pick the stronger direction between dir_ij and dir_ji using a deterministic score key priority:
      score_vis > score > alpha > e_raw
    Returns: (best_dir_dict, which="ij"|"ji")
    """
    dij = p.get("dir_ij") or {}
    dji = p.get("dir_ji") or {}

    def dir_strength(d: Dict[str, Any]) -> float:
        # prioritize what you actually trust; score_vis/score are usually already post-activation
        for k in ("score_vis", "score", "alpha", "e_raw"):
            if k in d and d[k] is not None:
                return _safe_float(d[k])
        return float("-inf")

    sij = dir_strength(dij)
    sji = dir_strength(dji)

    if sji > sij:
        return dji, "ji"
    return dij, "ij"


def _extract_af_pair_and_rel(p: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Extract (AF1, REL_TYPE, AF2) from a pair dict robustly.
    - Prefer best direction's src_text/dst_text and rel
    - Fallback to i_text/j_text if needed
    """
    best_dir, _ = _pick_best_direction(p)
    rel = str(best_dir.get("rel", "")).strip()

    af1 = str(best_dir.get("src_text", "")).strip()
    af2 = str(best_dir.get("dst_text", "")).strip()

    if not af1 or not af2:
        # fallback: these are typically always present
        af1 = str(p.get("i_text", "")).strip()
        af2 = str(p.get("j_text", "")).strip()

    return af1, rel, af2


def build_complementation_hints(
    selected_pairs: List[Dict[str, Any]],
    include_relation_defs: bool = True,
    include_bidir: bool = False,
) -> str:
    """
    Build structured hint block for LLM.

    Key changes vs your version:
    - REL_TYPE is NOT hardwired to dir_ij.rel anymore.
    - For each pair, choose the stronger direction (dir_ij vs dir_ji) based on score_vis/score/alpha/e_raw.
    - Optionally, include BOTH directions (include_bidir=True) to preserve asymmetric relations.
      (Default False: one best direction per pair to reduce prompt bloat.)

    Output format is deterministic and field-based to reduce LLM ambiguity.
    """
    lines: List[str] = []
    used_rels: set[str] = set()

    if not selected_pairs:
        # if include_relation_defs:
        #     lines.append("RELATION_TYPE_DEFINITIONS (one-line):")
        #     lines.append("- (none)")
        #     lines.append("")
        lines.append("STRONG_ATOMIC_FACT_LINKS (use as hints only):")
        lines.append("- (none)")
        return "\n".join(lines)

    # Collect used relation types (after direction selection)
    for p in selected_pairs:
        af1, rel, af2 = _extract_af_pair_and_rel(p)
        if af1 and af2 and rel:
            used_rels.add(rel)
        if include_bidir:
            # also add the opposite rel if it exists and differs
            dij = p.get("dir_ij") or {}
            dji = p.get("dir_ji") or {}
            r1 = str(dij.get("rel", "")).strip()
            r2 = str(dji.get("rel", "")).strip()
            if r1:
                used_rels.add(r1)
            if r2:
                used_rels.add(r2)

    # Relation definitions
    if include_relation_defs:
        defs = []
        for r in sorted([r for r in used_rels if r]):
            if r in REL_DEFS_ONE_LINER:
                defs.append(f"- {r}: {REL_DEFS_ONE_LINER[r]}")
        lines.append("RELATION_TYPE_DEFINITIONS (one-line):")
        lines.extend(defs if defs else ["- (none)"])
        lines.append("")

    # Links
    lines.append("STRONG_ATOMIC_FACT_LINKS (use as hints only):")

    link_idx = 1
    for p in selected_pairs:
        score = p.get("pair_score", None)
        score_str = f"{_safe_float(score, default=0.0):.4f}" if score is not None else "NA"

        if not include_bidir:
            af1, rel, af2 = _extract_af_pair_and_rel(p)
            if not af1 or not af2:
                continue

            rel_def = REL_DEFS_ONE_LINER.get(rel, "")
            lines.append(f"- Link {link_idx}:")            
            lines.append(f'  - AF1: "{af1}"')
            lines.append(f'  - REL_TYPE: "{rel}"')
            lines.append(f'  - REL_DEF: "{rel_def}"' if rel_def else '  - REL_DEF: ""')
            lines.append(f'  - AF2: "{af2}"')
            link_idx += 1
            continue

        # include_bidir=True: emit ij and ji separately (if present)
        dij = p.get("dir_ij") or {}
        dji = p.get("dir_ji") or {}

        for which, d in (("ij", dij), ("ji", dji)):
            rel = str(d.get("rel", "")).strip()
            af1 = str(d.get("src_text", "")).strip()
            af2 = str(d.get("dst_text", "")).strip()

            if not af1 or not af2:
                # fallback for missing src/dst text
                af1 = str(p.get("i_text", "")).strip()
                af2 = str(p.get("j_text", "")).strip()

            if not af1 or not af2:
                continue

            rel_def = REL_DEFS_ONE_LINER.get(rel, "")
            lines.append(f"- Link {link_idx}:")            
            lines.append(f'  - DIR: "{which}"')
            lines.append(f'  - AF1: "{af1}"')
            lines.append(f'  - REL_TYPE: "{rel}"')
            lines.append(f'  - REL_DEF: "{rel_def}"' if rel_def else '  - REL_DEF: ""')
            lines.append(f'  - AF2: "{af2}"')
            link_idx += 1

    # if all pairs were invalid (no AF texts), ensure non-empty section
    if link_idx == 1:
        lines.append("- (none)")

    return "\n".join(lines)


def triples_to_set(triples: List[List[str]]) -> set[Tuple[str, str, str]]:
    s: set[Tuple[str, str, str]] = set()
    for t in triples or []:
        if isinstance(t, list) and len(t) == 3:
            a, b, c = t
            if all(isinstance(x, str) for x in (a, b, c)):
                s.add((a.strip(), b.strip(), c.strip()))
    return s