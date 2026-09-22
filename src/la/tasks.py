"""Benchmark items: what gets asked, and what counts as right.

An item carries a prompt, an optional reference answer, and optional metadata
used for slicing results afterwards. Three loaders cover what real benchmarks
ship as: JSONL, CSV, and the Hugging Face hub.

Field names differ across every benchmark ever released, so loaders take a
mapping rather than assuming. Guessing column names is how a harness silently
evaluates the wrong field.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["Item", "load_jsonl", "load_csv_items", "load_hf", "slice_by"]


class Item:
    """One benchmark question."""

    __slots__ = ("id", "prompt", "reference", "metadata")

    def __init__(self, id: str, prompt: str, reference: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = str(id)
        self.prompt = str(prompt)
        self.reference = reference
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "prompt": self.prompt,
                "reference": self.reference, "metadata": self.metadata}

    def __repr__(self) -> str:
        head = self.prompt[:48].replace("\n", " ")
        return "Item(%s, %r%s)" % (self.id, head,
                                   "..." if len(self.prompt) > 48 else "")


def _build(
    records: Iterable[Dict[str, Any]],
    prompt_field: str,
    reference_field: Optional[str],
    id_field: Optional[str],
    metadata_fields: Optional[Sequence[str]],
    limit: Optional[int],
    source: str,
) -> List[Item]:
    items: List[Item] = []
    for index, row in enumerate(records):
        if limit is not None and len(items) >= limit:
            break
        if prompt_field not in row:
            raise ValueError(
                "%s row %d has no field %r. Available: %s"
                % (source, index, prompt_field, ", ".join(sorted(row))[:200])
            )
        item_id = str(row[id_field]) if id_field and id_field in row \
            else "item_%05d" % index
        reference = None
        if reference_field and reference_field in row:
            reference = str(row[reference_field])
        metadata = {}
        if metadata_fields:
            metadata = {f: row.get(f) for f in metadata_fields if f in row}
        items.append(Item(item_id, str(row[prompt_field]), reference, metadata))

    if not items:
        raise ValueError("%s produced no items" % source)
    _check_unique(items, source)
    return items


def _check_unique(items: Sequence[Item], source: str) -> None:
    seen = set()
    for item in items:
        if item.id in seen:
            raise ValueError(
                "%s has duplicate item id %r. Item ids are the pairing key "
                "across systems; duplicates would silently merge two "
                "questions into one cell." % (source, item.id)
            )
        seen.add(item.id)


def load_jsonl(
    path: str,
    prompt_field: str = "question",
    reference_field: Optional[str] = "answer",
    id_field: Optional[str] = None,
    metadata_fields: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Item]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("%s line %d is not valid JSON: %s"
                                 % (path, line_no, exc))
    return _build(records, prompt_field, reference_field, id_field,
                  metadata_fields, limit, os.path.basename(path))


def load_csv_items(
    path: str,
    prompt_field: str = "question",
    reference_field: Optional[str] = "answer",
    id_field: Optional[str] = None,
    metadata_fields: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
) -> List[Item]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, newline="", encoding="utf-8") as fh:
        records = [dict(row) for row in csv.DictReader(fh)]
    return _build(records, prompt_field, reference_field, id_field,
                  metadata_fields, limit, os.path.basename(path))


def load_hf(
    dataset: str,
    split: str = "test",
    prompt_field: str = "question",
    reference_field: Optional[str] = "answer",
    id_field: Optional[str] = None,
    metadata_fields: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    config: Optional[str] = None,
) -> List[Item]:
    """Load from the Hugging Face hub.

    `datasets` is an optional dependency; the message says so rather than
    letting an ImportError surface from three frames down.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "loading from the Hugging Face hub needs the `datasets` package: "
            "pip install datasets. JSONL and CSV loaders need nothing."
        )
    data = load_dataset(dataset, config, split=split) if config \
        else load_dataset(dataset, split=split)
    if limit is not None:
        data = data.select(range(min(limit, len(data))))
    return _build((dict(row) for row in data), prompt_field, reference_field,
                  id_field, metadata_fields, limit, dataset)


def slice_by(items: Sequence[Item], field: str) -> Dict[str, List[str]]:
    """Group item ids by a metadata field.

    An overall tier count can hide that a system leads on one slice and trails
    on another, so the audit can be re-run per slice.
    """
    out: Dict[str, List[str]] = {}
    for item in items:
        key = str(item.metadata.get(field, "unknown"))
        out.setdefault(key, []).append(item.id)
    return out
