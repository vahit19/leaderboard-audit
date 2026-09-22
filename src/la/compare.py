"""Comparing two graders on the same answers.

Noise and bias are different problems and only one of them is usually looked
for. A grader that disagrees with itself is noisy, and repeat grading finds it.
A grader that is perfectly repeatable can still be wrong in a way that depends
on which system produced the answer -- and that is invisible to any amount of
repeat grading, because every repeat is wrong in the same direction.

Per-system bias is the dangerous kind. If a grader is 3 points generous to one
system and 6 points harsh to another, it has moved them 9 points apart, and on
a saturated benchmark where real gaps are 1-2 points that is enough to reorder
the table. The headline number, overall agreement, hides this completely: a
grader can be 94% consistent overall and still invert the ranking.

This module takes two score tables over the same systems and items and reports
the disagreement per system, what it does to the ordering, and whether the two
graders would have produced different rankings.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .data import ResultTable
from .stats.exact import paired_summary

__all__ = ["align", "grader_bias", "ranking_changes", "compare_graders"]


def align(
    reference: ResultTable,
    candidate: ResultTable,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Restrict two tables to the systems and items they share.

    Returns (systems, items, reference_scores, candidate_scores), both arrays
    shaped (n_items, n_systems). Comparing graders on different item sets would
    confound the grader with the sample, which is the exact mistake this is
    meant to catch elsewhere.
    """
    systems = sorted(set(reference.systems) & set(candidate.systems))
    items = sorted(set(reference.items) & set(candidate.items))
    if not systems or not items:
        raise ValueError(
            "the two tables share %d system(s) and %d item(s); there is "
            "nothing to compare" % (len(systems), len(items))
        )

    def extract(table: ResultTable) -> np.ndarray:
        s_index = {s: j for j, s in enumerate(table.systems)}
        i_index = {it: i for i, it in enumerate(table.items)}
        out = np.full((len(items), len(systems)), np.nan)
        for i, item in enumerate(items):
            for j, system in enumerate(systems):
                out[i, j] = table.scores[i_index[item], s_index[system]]
        return out

    ref = extract(reference)
    cand = extract(candidate)
    keep = ~(np.isnan(ref).any(axis=1) | np.isnan(cand).any(axis=1))
    return systems, [items[i] for i in np.where(keep)[0]], ref[keep], cand[keep]


def grader_bias(
    systems: Sequence[str],
    reference: np.ndarray,
    candidate: np.ndarray,
    alpha: float = 0.05,
    seed: int = 0,
) -> List[Dict[str, object]]:
    """Per-system difference between the two graders, paired over items.

    Paired, because both graders saw the same answers: an item that is hard to
    grade is hard for both, and pairing removes that shared difficulty instead
    of counting it as disagreement.
    """
    rows: List[Dict[str, object]] = []
    for j, system in enumerate(systems):
        ref_col, cand_col = reference[:, j], candidate[:, j]
        row = paired_summary(system, cand_col, ref_col, seed=seed)
        row["system"] = system
        row["reference_score"] = float(np.mean(ref_col))
        row["candidate_score"] = float(np.mean(cand_col))
        row["bias"] = row.pop("delta")
        row["disagreement_rate"] = float(np.mean(
            (cand_col >= 0.5).astype(float) != (ref_col >= 0.5).astype(float)
        ))
        row["too_generous"] = int(np.sum(
            (cand_col >= 0.5) & (ref_col < 0.5)))
        row["too_harsh"] = int(np.sum(
            (cand_col < 0.5) & (ref_col >= 0.5)))
        rows.append(row)

    if rows:
        from .stats.exact import holm
        adjusted = holm([float(r["p_value"]) for r in rows],
                        [str(r["system"]) for r in rows])
        for row, (_, _, adj) in zip(rows, adjusted):
            row["holm_p"] = adj
            row["biased"] = bool(adj < alpha)
    rows.sort(key=lambda r: r["bias"])
    return rows


def ranking_changes(
    systems: Sequence[str],
    reference: np.ndarray,
    candidate: np.ndarray,
) -> Dict[str, object]:
    """What swapping graders does to the ordering."""
    ref_means = np.nanmean(reference, axis=0)
    cand_means = np.nanmean(candidate, axis=0)
    ref_rank = _ranks_desc(ref_means)
    cand_rank = _ranks_desc(cand_means)

    moves = []
    for j, system in enumerate(systems):
        if ref_rank[j] != cand_rank[j]:
            moves.append({
                "system": system,
                "reference_rank": int(ref_rank[j]),
                "candidate_rank": int(cand_rank[j]),
                "moved": int(cand_rank[j] - ref_rank[j]),
            })

    inversions = []
    for a in range(len(systems)):
        for b in range(a + 1, len(systems)):
            ref_order = np.sign(ref_means[a] - ref_means[b])
            cand_order = np.sign(cand_means[a] - cand_means[b])
            if ref_order != 0 and cand_order != 0 and ref_order != cand_order:
                better = systems[a] if ref_order > 0 else systems[b]
                worse = systems[b] if ref_order > 0 else systems[a]
                inversions.append({"reference_better": better,
                                   "candidate_better": worse})

    return {
        "moved": moves,
        "inversions": inversions,
        "n_moved": len(moves),
        "n_inversions": len(inversions),
        "top_changed": bool(systems[int(np.argmax(ref_means))]
                            != systems[int(np.argmax(cand_means))]),
    }


def compare_graders(
    reference: ResultTable,
    candidate: ResultTable,
    reference_name: str = "reference grader",
    candidate_name: str = "candidate grader",
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, object]:
    """Full comparison of two graders over the same answers."""
    systems, items, ref, cand = align(reference, candidate)
    bias = grader_bias(systems, ref, cand, alpha=alpha, seed=seed)
    changes = ranking_changes(systems, ref, cand)

    overall = float(np.mean(
        (cand >= 0.5).astype(float) == (ref >= 0.5).astype(float)))
    spread = (max(r["bias"] for r in bias) - min(r["bias"] for r in bias)
              if bias else 0.0)

    return {
        "reference_name": reference_name,
        "candidate_name": candidate_name,
        "n_systems": len(systems),
        "n_items": len(items),
        "overall_agreement": overall,
        "bias_spread": spread,
        "per_system": bias,
        "ranking": changes,
        "verdict": _verdict(overall, spread, changes, bias),
    }


def _verdict(overall: float, spread: float, changes: Dict[str, object],
             bias: Sequence[Dict[str, object]]) -> str:
    biased = [r for r in bias if r.get("biased")]
    parts = ["overall agreement %.1f%%" % (100 * overall)]

    if spread > 0:
        parts.append("but the grader's error is not even across systems: it "
                     "spans %.1f points between the system it treats best and "
                     "the one it treats worst" % (100 * spread))
    if biased:
        parts.append("%d system(s) show a bias that survives correction" %
                     len(biased))
    if changes["n_inversions"]:
        parts.append("and the two graders disagree on the order of %d pair(s)"
                     % changes["n_inversions"])
    if changes["top_changed"]:
        parts.append("including which system is first")

    tail = (". On a saturated benchmark, a bias spread larger than the real "
            "gaps between systems is enough to reorder the table, and no "
            "amount of repeat grading would reveal it -- every repeat is "
            "wrong in the same direction."
            if spread > 0.02 and changes["n_inversions"] else ".")
    return "; ".join(parts) + tail


def _ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1, dtype=float)
    return ranks
