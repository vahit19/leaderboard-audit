"""Spend accounting and a hard stop.

A measurement run is a loop that calls a paid API thousands of times. The
failure mode is not a crash, it is a loop that works perfectly and bills all
night. Every run therefore carries a ceiling it cannot cross, and the ceiling
is checked before each call rather than after.

Prices are supplied by the caller or read from the provider's own model list.
Nothing here hardcodes a price, because a hardcoded price silently becomes
wrong and a wrong price in a budget check is worse than no check.
"""
from __future__ import annotations

import threading
from typing import Dict, Optional

__all__ = ["BudgetExceeded", "Budget", "Price"]


class BudgetExceeded(RuntimeError):
    """Raised when a call would cross the ceiling. The run stops, it does not
    silently degrade: partial results with an unexplained gap are worse than a
    clear halt, and the cache means resuming costs nothing."""


class Price:
    """Cost per million tokens, split by direction."""

    __slots__ = ("prompt", "completion")

    def __init__(self, prompt: float, completion: float):
        self.prompt = float(prompt)
        self.completion = float(completion)

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (prompt_tokens * self.prompt / 1e6
                + completion_tokens * self.completion / 1e6)

    def __repr__(self) -> str:
        return "Price(in=$%.3f/M, out=$%.3f/M)" % (self.prompt, self.completion)


class Budget:
    """Running spend against a hard ceiling, safe across threads."""

    def __init__(self, limit_usd: Optional[float] = None):
        self.limit_usd = float(limit_usd) if limit_usd is not None else None
        self.spent_usd = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0
        self.cached_calls = 0
        self._lock = threading.Lock()

    def check(self, projected_usd: float = 0.0) -> None:
        """Refuse a call that would cross the ceiling."""
        if self.limit_usd is None:
            return
        with self._lock:
            if self.spent_usd + projected_usd > self.limit_usd:
                raise BudgetExceeded(
                    "spend limit reached: $%.4f spent, this call adds about "
                    "$%.4f, limit is $%.2f. Raise --budget to continue; "
                    "completed work is cached and will not be paid for again."
                    % (self.spent_usd, projected_usd, self.limit_usd)
                )

    def record(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        price: Optional[Price] = None,
        cost_usd: Optional[float] = None,
        cached: bool = False,
    ) -> float:
        """Add one call. A cached call costs nothing but is still counted."""
        if cached:
            with self._lock:
                self.cached_calls += 1
            return 0.0

        if cost_usd is None:
            cost_usd = price.cost(prompt_tokens, completion_tokens) if price else 0.0

        with self._lock:
            self.calls += 1
            self.prompt_tokens += int(prompt_tokens)
            self.completion_tokens += int(completion_tokens)
            self.spent_usd += float(cost_usd)
        return float(cost_usd)

    def remaining(self) -> Optional[float]:
        if self.limit_usd is None:
            return None
        with self._lock:
            return max(0.0, self.limit_usd - self.spent_usd)

    def summary(self) -> Dict[str, object]:
        with self._lock:
            return {
                "spent_usd": round(self.spent_usd, 4),
                "limit_usd": self.limit_usd,
                "billed_calls": self.calls,
                "cached_calls": self.cached_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }

    def __repr__(self) -> str:
        limit = "none" if self.limit_usd is None else "$%.2f" % self.limit_usd
        return "Budget(spent=$%.4f, limit=%s)" % (self.spent_usd, limit)


def estimate_tokens(text: str) -> int:
    """Rough token count for pre-flight estimates only.

    Deliberately crude: about four characters per token. It is used to project
    a cost before a run and to reserve budget before a call, never to bill.
    Actual usage always comes back from the provider.
    """
    return max(1, len(text) // 4)
