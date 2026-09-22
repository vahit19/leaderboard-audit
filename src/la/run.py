"""The measurement run: ask every system every item, repeatedly, and grade.

A run is a loop over (item, system, repeat). Three properties make it usable
on real work rather than on a demo:

  Resumable. Every finished cell is appended to results.jsonl the moment it
  lands. Re-running skips what is already there, so a run killed at item 400
  of 600 costs nothing to restart and a crash never loses paid work.

  Bounded. The budget is checked before each call, not after. A loop that works
  perfectly and bills all night is the failure mode that matters.

  Honest about failure. A cell that errors is recorded with its error and
  excluded from the table, and the count is reported. Silently dropping failed
  cells biases results toward whichever systems fail least on hard items --
  exactly the items that separate systems.

Answers and grades are separate calls with separate repeat indices, so
generation variance and grading variance stay distinguishable. Collapsing them
would make the whole variance decomposition meaningless.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .budget import Budget, BudgetExceeded
from .judge import Score, Scorer
from .providers import Provider, ProviderError
from .tasks import Item

__all__ = ["RunConfig", "RunResult", "run_evaluation", "load_results",
           "write_long_csv"]

DEFAULT_SYSTEM_PROMPT = (
    "Answer the question. Be direct and give only the answer."
)


class RunConfig:
    """Everything one measurement run needs."""

    def __init__(
        self,
        systems: Sequence[str],
        items: Sequence[Item],
        scorer: Scorer,
        answer_repeats: int = 1,
        grade_repeats: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        concurrency: int = 4,
        out_dir: str = "run",
    ):
        if answer_repeats < 1 or grade_repeats < 1:
            raise ValueError("repeat counts must be at least 1")
        if answer_repeats > 1 and temperature == 0.0:
            raise ValueError(
                "answer_repeats > 1 at temperature 0 measures nothing: the "
                "model is being asked to repeat itself deterministically. "
                "Raise the temperature, or set answer_repeats back to 1 and "
                "use grade_repeats to measure the judge instead."
            )
        self.systems = list(systems)
        self.items = list(items)
        self.scorer = scorer
        self.answer_repeats = int(answer_repeats)
        self.grade_repeats = int(grade_repeats)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.system_prompt = system_prompt
        self.concurrency = max(1, int(concurrency))
        self.out_dir = out_dir

    @property
    def total_cells(self) -> int:
        return (len(self.systems) * len(self.items)
                * self.answer_repeats * self.grade_repeats)

    def describe(self) -> Dict[str, Any]:
        return {
            "systems": self.systems,
            "n_items": len(self.items),
            "answer_repeats": self.answer_repeats,
            "grade_repeats": self.grade_repeats,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "scorer": getattr(self.scorer, "name", type(self.scorer).__name__),
            "concurrency": self.concurrency,
            "total_cells": self.total_cells,
        }


class RunResult:
    def __init__(self, rows: List[Dict[str, Any]], errors: List[Dict[str, Any]],
                 budget: Budget, config: RunConfig, seconds: float,
                 stopped_early: bool = False, stop_reason: str = ""):
        self.rows = rows
        self.errors = errors
        self.budget = budget
        self.config = config
        self.seconds = seconds
        self.stopped_early = stopped_early
        self.stop_reason = stop_reason

    def summary(self) -> Dict[str, Any]:
        unparsed = sum(1 for r in self.rows if not r.get("parsed", True))
        return {
            "rows": len(self.rows),
            "errors": len(self.errors),
            "unparsed_grades": unparsed,
            "expected_cells": self.config.total_cells,
            "seconds": round(self.seconds, 1),
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "budget": self.budget.summary(),
        }


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

def _cell_id(system: str, item_id: str, answer_rep: int, grade_rep: int) -> str:
    return "%s|%s|a%d|g%d" % (system, item_id, answer_rep, grade_rep)


def load_results(path: str) -> Dict[str, Dict[str, Any]]:
    """Read results.jsonl into a map keyed by cell, for resuming.

    A truncated final line from a killed process is skipped rather than
    treated as corruption; that line's cell simply gets recomputed.
    """
    if not os.path.exists(path):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            out[row["cell"]] = row
    return out


class _Appender:
    """Append-only writer, flushed per row so a kill loses nothing."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, row: Dict[str, Any]) -> None:
        with self._lock:
            self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            self._fh.close()


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------

def run_evaluation(
    config: RunConfig,
    provider: Provider,
    budget: Optional[Budget] = None,
    resume: bool = True,
    progress: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> RunResult:
    """Execute the run, resuming from whatever is already on disk."""
    started = time.time()
    budget = budget or getattr(provider, "budget", None) or Budget()

    os.makedirs(config.out_dir, exist_ok=True)
    results_path = os.path.join(config.out_dir, "results.jsonl")
    errors_path = os.path.join(config.out_dir, "errors.jsonl")

    done = load_results(results_path) if resume else {}
    if not resume and os.path.exists(results_path):
        os.replace(results_path, results_path + ".bak")

    work = _plan(config, done)
    rows = list(done.values())
    errors: List[Dict[str, Any]] = []

    if not work:
        return RunResult(rows, errors, budget, config, time.time() - started)

    writer = _Appender(results_path)
    error_writer = _Appender(errors_path)
    stop = threading.Event()
    stop_reason = [""]
    completed = [0]
    lock = threading.Lock()

    def execute(unit) -> None:
        if stop.is_set():
            return
        system, item, answer_rep, grade_rep = unit
        cell = _cell_id(system, item.id, answer_rep, grade_rep)
        try:
            row = _one_cell(config, provider, system, item, answer_rep,
                            grade_rep, cell)
        except BudgetExceeded as exc:
            stop.set()
            stop_reason[0] = str(exc)
            return
        except (ProviderError, Exception) as exc:  # noqa: BLE001
            record = {"cell": cell, "system": system, "item": item.id,
                      "answer_repeat": answer_rep, "grade_repeat": grade_rep,
                      "error": "%s: %s" % (type(exc).__name__, exc)}
            error_writer.write(record)
            with lock:
                errors.append(record)
                completed[0] += 1
            return

        writer.write(row)
        with lock:
            rows.append(row)
            completed[0] += 1
            if progress:
                progress(completed[0], len(work), row)

    try:
        with ThreadPoolExecutor(max_workers=config.concurrency) as pool:
            futures = [pool.submit(execute, unit) for unit in work]
            for future in as_completed(futures):
                future.result()
    finally:
        writer.close()
        error_writer.close()

    return RunResult(rows, errors, budget, config, time.time() - started,
                     stopped_early=stop.is_set(), stop_reason=stop_reason[0])


def _plan(config: RunConfig, done: Dict[str, Dict[str, Any]]):
    """Units of work not already on disk.

    Ordered item-major so that an interrupted run still covers every system on
    the items it reached. A system-major order would leave the table ragged --
    the first systems complete, the last ones empty -- and a ragged table
    cannot be paired.
    """
    units = []
    for item in config.items:
        for answer_rep in range(1, config.answer_repeats + 1):
            for system in config.systems:
                for grade_rep in range(1, config.grade_repeats + 1):
                    cell = _cell_id(system, item.id, answer_rep, grade_rep)
                    if cell not in done:
                        units.append((system, item, answer_rep, grade_rep))
    return units


def _one_cell(config: RunConfig, provider: Provider, system: str, item: Item,
              answer_rep: int, grade_rep: int, cell: str) -> Dict[str, Any]:
    """Generate one answer and grade it once."""
    messages = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": item.prompt})

    completion = provider.complete(
        messages, model=system, temperature=config.temperature,
        max_tokens=config.max_tokens, sample=answer_rep,
    )

    score: Score = config.scorer.score(
        item.prompt, completion.text, item.reference, sample=grade_rep,
    )

    # A cached answer was paid for by an earlier run, so it bills nothing now.
    # Both numbers are kept: `cost_usd` sums to what this run actually spent,
    # `cell_cost_usd` to what producing the cell cost the first time. Reporting
    # only the second would make a resumed run look as expensive as the
    # original, which is the opposite of the truth.
    billed_answer = 0.0 if completion.cached else completion.cost_usd

    return {
        "cell": cell,
        "system": system,
        "item": item.id,
        "answer_repeat": answer_rep,
        "grade_repeat": grade_rep,
        "score": score.value,
        "grader": score.grader,
        "parsed": score.parsed,
        "answer": completion.text[:2000],
        "answer_cached": completion.cached,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "cost_usd": round(billed_answer + score.cost_usd, 6),
        "cell_cost_usd": round(completion.cost_usd + score.cost_usd, 6),
        "metadata": item.metadata,
    }


# ---------------------------------------------------------------------------
# handing the run to the audit
# ---------------------------------------------------------------------------

def write_long_csv(rows: Iterable[Dict[str, Any]], path: str,
                   drop_unparsed: bool = False) -> Tuple[int, int]:
    """Write the long table the audit reads: system,item,run,score.

    The `run` column combines the answer and grade repeat indices, so every
    repeat stays a separate row and the variance decomposition has something
    to work with.

    Returns (written, skipped).
    """
    import csv

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = skipped = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["system", "item", "run", "score"])
        for row in rows:
            if "score" not in row:
                skipped += 1
                continue
            if drop_unparsed and not row.get("parsed", True):
                skipped += 1
                continue
            run_id = "a%d_g%d" % (row.get("answer_repeat", 1),
                                  row.get("grade_repeat", 1))
            writer.writerow([row["system"], row["item"], run_id, row["score"]])
            written += 1
    return written, skipped


def console_progress(every: int = 10):
    """Progress callback that overwrites one line and stays quiet otherwise."""
    state = {"last": 0.0}

    def report(done: int, total: int, row: Dict[str, Any]) -> None:
        now = time.time()
        if done % every and now - state["last"] < 2.0 and done != total:
            return
        state["last"] = now
        pct = 100.0 * done / max(1, total)
        sys.stderr.write("\r  %d/%d cells (%.0f%%)  " % (done, total, pct))
        sys.stderr.flush()
        if done == total:
            sys.stderr.write("\n")

    return report
