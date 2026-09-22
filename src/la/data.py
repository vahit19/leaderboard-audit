"""Loading evaluation results into the one shape everything else expects.

The input is a long table with three required columns and one optional one:

    system,item,score[,run]

`item` is whatever the benchmark has a finite number of -- a task, a prompt, a
sample, a conversation. `run` is present when the same system was evaluated on
the same item more than once, which is what makes measurement noise separable
from system differences.

Long format is deliberate. Published leaderboards are wide -- one row per
system, one aggregate number -- and that shape has already thrown away
everything needed to say whether the ordering means anything. If only the wide
table exists, the audit cannot be done, and that itself is worth reporting.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ResultTable", "load_csv", "from_records"]

REQUIRED = ("system", "item", "score")


class ResultTable:
    """A complete system-by-item score matrix, plus optional repeat runs."""

    def __init__(
        self,
        systems: Sequence[str],
        items: Sequence[str],
        scores: np.ndarray,
        repeats: Optional[Dict[Tuple[str, str], List[float]]] = None,
        source: str = "",
    ):
        self.systems = list(systems)
        self.items = list(items)
        self.scores = np.asarray(scores, dtype=float)   # (n_items, n_systems)
        self.repeats = repeats or {}
        self.source = source

        if self.scores.shape != (len(self.items), len(self.systems)):
            raise ValueError(
                "scores must be (n_items, n_systems); got %s for %d items and "
                "%d systems" % (self.scores.shape, len(self.items),
                                len(self.systems))
            )

    # -- shape ------------------------------------------------------------

    @property
    def n_items(self) -> int:
        return len(self.items)

    @property
    def n_systems(self) -> int:
        return len(self.systems)

    @property
    def has_repeats(self) -> bool:
        return any(len(v) > 1 for v in self.repeats.values())

    @property
    def n_missing(self) -> int:
        return int(np.isnan(self.scores).sum())

    def complete_items(self) -> np.ndarray:
        """Indices of items every system was evaluated on.

        Pairing only holds on these. Dropping the rest is the conservative
        choice and the count is reported so the reader knows what it cost.
        """
        return np.where(~np.isnan(self.scores).any(axis=1))[0]

    def complete(self) -> "ResultTable":
        keep = self.complete_items()
        return ResultTable(
            self.systems,
            [self.items[i] for i in keep],
            self.scores[keep],
            self.repeats,
            self.source,
        )

    def describe(self) -> Dict[str, object]:
        complete = self.complete_items()
        repeat_counts = [len(v) for v in self.repeats.values()] or [1]
        return {
            "source": self.source,
            "systems": self.n_systems,
            "items": self.n_items,
            "complete_items": int(complete.size),
            "missing_cells": self.n_missing,
            "has_repeats": self.has_repeats,
            "min_repeats": int(min(repeat_counts)),
            "max_repeats": int(max(repeat_counts)),
        }


def from_records(
    records: Sequence[Dict[str, object]],
    source: str = "",
) -> ResultTable:
    """Build a table from dict rows with keys system/item/score[/run]."""
    if not records:
        raise ValueError("no records")

    missing = [c for c in REQUIRED if c not in records[0]]
    if missing:
        raise ValueError("missing required column(s): %s" % ", ".join(missing))

    systems: List[str] = []
    items: List[str] = []
    seen_systems, seen_items = set(), set()
    cells: Dict[Tuple[str, str], List[float]] = {}

    for row in records:
        s, it = str(row["system"]), str(row["item"])
        try:
            value = float(row["score"])
        except (TypeError, ValueError):
            raise ValueError("score is not numeric for system=%s item=%s" % (s, it))
        if s not in seen_systems:
            seen_systems.add(s)
            systems.append(s)
        if it not in seen_items:
            seen_items.add(it)
            items.append(it)
        cells.setdefault((s, it), []).append(value)

    systems.sort()
    items.sort()
    s_index = {s: j for j, s in enumerate(systems)}
    i_index = {it: i for i, it in enumerate(items)}

    scores = np.full((len(items), len(systems)), np.nan, dtype=float)
    for (s, it), values in cells.items():
        scores[i_index[it], s_index[s]] = float(np.mean(values))

    return ResultTable(systems, items, scores, cells, source)


def load_csv(path: str) -> ResultTable:
    """Read a long-format CSV. Raises with a usable message on bad input."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError("%s has no header row" % path)
        missing = [c for c in REQUIRED if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                "%s is missing column(s): %s. Expected a long table with "
                "%s[,run]." % (path, ", ".join(missing), ",".join(REQUIRED))
            )
        records = [dict(row) for row in reader]
    if not records:
        raise ValueError("%s has a header but no rows" % path)
    return from_records(records, source=os.path.basename(path))
