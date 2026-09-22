"""Model providers.

Three of them, and the third is the reason the other two are written this way:

  OpenRouterProvider  real calls, with retries, live pricing and budget checks.
  ReplayProvider      serves only cache hits. No key, no network, no spend.
  StubProvider        deterministic fake answers, for tests.

`ReplayProvider` is what makes a published run checkable. Ship the cache next
to the results and a reader re-executes the identical pipeline offline; if a
call is missing from the cache it raises instead of quietly going to the
network, so a "reproduction" cannot silently become a fresh paid run.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional

from .budget import Budget, BudgetExceeded, Price, estimate_tokens
from .cache import CallKey, DiskCache

__all__ = ["Completion", "Provider", "OpenRouterProvider", "ReplayProvider",
           "StubProvider", "ProviderError"]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class ProviderError(RuntimeError):
    pass


class Completion:
    """One model response plus what it cost."""

    __slots__ = ("text", "model", "prompt_tokens", "completion_tokens",
                 "cost_usd", "cached", "raw")

    def __init__(self, text: str, model: str, prompt_tokens: int = 0,
                 completion_tokens: int = 0, cost_usd: float = 0.0,
                 cached: bool = False, raw: Optional[Dict[str, Any]] = None):
        self.text = text
        self.model = model
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        self.cost_usd = float(cost_usd)
        self.cached = bool(cached)
        self.raw = raw or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": self.cost_usd}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Completion":
        return cls(d["text"], d.get("model", ""), d.get("prompt_tokens", 0),
                   d.get("completion_tokens", 0), d.get("cost_usd", 0.0),
                   cached=True)


class Provider:
    """Interface every provider implements."""

    name = "provider"

    def complete(self, messages: List[Dict[str, str]], model: str,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 sample: int = 0) -> Completion:
        raise NotImplementedError

    def price_for(self, model: str) -> Optional[Price]:
        return None


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------

class OpenRouterProvider(Provider):
    """OpenRouter chat completions with caching, retries and a spend ceiling.

    Retries only what is worth retrying: 429 and 5xx, with exponential backoff
    and jitter. A 400 is a bug in the request and retrying it just burns time;
    a 401 is a bad key and will not fix itself.
    """

    name = "openrouter"
    RETRY_STATUS = (408, 409, 429, 500, 502, 503, 504)

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[DiskCache] = None,
        budget: Optional[Budget] = None,
        max_retries: int = 5,
        timeout: float = 120.0,
        referer: str = "https://github.com/leaderboard-audit",
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY") \
            or os.environ.get("openrouter_api_key")
        if not self.api_key:
            raise ProviderError(
                "no OpenRouter key. Set OPENROUTER_API_KEY in the environment "
                "or a .env file. The key is read from the environment and is "
                "never written to results, logs or the cache."
            )
        self.cache = cache or DiskCache(".cache", enabled=True)
        self.budget = budget or Budget()
        self.max_retries = int(max_retries)
        self.timeout = float(timeout)
        self.referer = referer
        self._prices: Optional[Dict[str, Price]] = None
        self._price_lock = threading.Lock()

    # -- pricing ------------------------------------------------------------

    def price_for(self, model: str) -> Optional[Price]:
        """Live prices from the provider, fetched once and reused.

        Read from the API rather than hardcoded: a stale hardcoded price makes
        a budget ceiling wrong in a direction nobody notices until the bill.
        """
        with self._price_lock:
            if self._prices is None:
                self._prices = self._fetch_prices()
        return self._prices.get(model)

    def _fetch_prices(self) -> Dict[str, Price]:
        import requests
        try:
            response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
            response.raise_for_status()
            data = response.json().get("data", [])
        except Exception:
            return {}
        out: Dict[str, Price] = {}
        for entry in data:
            pricing = entry.get("pricing") or {}
            try:
                # OpenRouter quotes per token; this module works per million.
                out[entry["id"]] = Price(
                    float(pricing.get("prompt", 0.0)) * 1e6,
                    float(pricing.get("completion", 0.0)) * 1e6,
                )
            except (TypeError, ValueError, KeyError):
                continue
        return out

    # -- completion ---------------------------------------------------------

    def complete(self, messages: List[Dict[str, str]], model: str,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 sample: int = 0) -> Completion:
        key = CallKey(model, messages, temperature, max_tokens, sample)

        hit = self.cache.get(key)
        if hit is not None:
            completion = Completion.from_dict(hit)
            self.budget.record(0, 0, cached=True)
            return completion

        # Reserve an estimate before spending, so the ceiling binds in advance.
        prompt_text = "".join(m.get("content", "") for m in messages)
        price = self.price_for(model)
        projected = 0.0
        if price is not None:
            projected = price.cost(estimate_tokens(prompt_text), max_tokens)
        self.budget.check(projected)

        payload = self._request(messages, model, temperature, max_tokens)

        choice = (payload.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))

        # OpenRouter reports actual cost on the response when available; prefer
        # it over our own multiplication.
        reported = usage.get("cost")
        cost = self.budget.record(
            prompt_tokens, completion_tokens, price=price,
            cost_usd=float(reported) if reported is not None else None,
        )

        completion = Completion(text, model, prompt_tokens, completion_tokens,
                                cost, cached=False, raw=payload)
        self.cache.put(key, completion.to_dict())
        return completion

    def _request(self, messages, model, temperature, max_tokens) -> Dict[str, Any]:
        import requests

        headers = {
            "Authorization": "Bearer %s" % self.api_key,
            "Content-Type": "application/json",
            "HTTP-Referer": self.referer,
            "X-Title": "leaderboard-audit",
        }
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }

        last_error = ""
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(OPENROUTER_URL, headers=headers,
                                         json=body, timeout=self.timeout)
            except Exception as exc:                      # network-level
                last_error = "%s: %s" % (type(exc).__name__, exc)
                self._sleep(attempt)
                continue

            if response.status_code == 200:
                try:
                    payload = response.json()
                except ValueError:
                    last_error = "200 with a body that is not JSON"
                    self._sleep(attempt)
                    continue
                if payload.get("error"):
                    # OpenRouter can return 200 carrying an upstream error.
                    last_error = json.dumps(payload["error"])[:300]
                    self._sleep(attempt)
                    continue
                return payload

            if response.status_code in (401, 403):
                raise ProviderError(
                    "OpenRouter rejected the key (HTTP %d). Check "
                    "OPENROUTER_API_KEY." % response.status_code
                )
            if response.status_code not in self.RETRY_STATUS:
                raise ProviderError(
                    "OpenRouter returned HTTP %d: %s"
                    % (response.status_code, response.text[:300])
                )

            last_error = "HTTP %d: %s" % (response.status_code,
                                          response.text[:200])
            self._sleep(attempt, response.headers.get("Retry-After"))

        raise ProviderError(
            "gave up on %s after %d attempts. Last error: %s"
            % (model, self.max_retries + 1, last_error)
        )

    def _sleep(self, attempt: int, retry_after: Optional[str] = None) -> None:
        if retry_after:
            try:
                time.sleep(min(60.0, float(retry_after)))
                return
            except (TypeError, ValueError):
                pass
        # Exponential with jitter, so a fleet of workers does not retry in step.
        delay = min(30.0, (2.0 ** attempt)) * (0.5 + random.random())
        time.sleep(delay)


# ---------------------------------------------------------------------------
# offline
# ---------------------------------------------------------------------------

class ReplayProvider(Provider):
    """Serves cached calls only. Raises on a miss rather than going online."""

    name = "replay"

    def __init__(self, cache: DiskCache):
        self.cache = cache
        self.budget = Budget()

    def complete(self, messages: List[Dict[str, str]], model: str,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 sample: int = 0) -> Completion:
        key = CallKey(model, messages, temperature, max_tokens, sample)
        hit = self.cache.get(key)
        if hit is None:
            raise ProviderError(
                "replay miss for %s sample %d (%s). The cache does not cover "
                "this run, and replay will not go to the network. Either "
                "obtain the full cache or re-run with a live provider."
                % (model, sample, key.digest()[:12])
            )
        self.budget.record(0, 0, cached=True)
        return Completion.from_dict(hit)


STUB_MARKER = "STUBANSWER"


class StubProvider(Provider):
    """Simulates both the systems under test and a noisy judge.

    It exists so the whole pipeline -- generation, grading, repeats, resume,
    audit -- can be exercised end to end with no key, no network and no spend,
    and so the audit's output can be checked against a truth that was set on
    purpose.

    Each model id carries a stable true accuracy, taken from `accuracies` or
    derived from the name. Answering emits a deterministic token carrying the
    model's identity; grading recovers that identity and returns CORRECT with
    that model's accuracy, perturbed per grading sample by `judge_noise`.

    That perturbation is the point. With `judge_noise` above zero, grading the
    same answer twice can disagree -- which is the behaviour this whole system
    was built to quantify, and a simulator that graded deterministically would
    give the variance decomposition nothing to find.
    """

    name = "stub"

    def __init__(
        self,
        accuracy: float = 0.7,
        noise: float = 0.0,
        judge_noise: Optional[float] = None,
        accuracies: Optional[Dict[str, float]] = None,
        seed: int = 0,
    ):
        self.accuracy = float(accuracy)
        self.noise = float(noise)
        self.judge_noise = float(noise if judge_noise is None else judge_noise)
        self.accuracies = dict(accuracies or {})
        self.seed = int(seed)
        self.budget = Budget()
        self.calls = 0

    # -- per-model truth ----------------------------------------------------

    def true_accuracy(self, model: str) -> float:
        """Stable true accuracy for a model id.

        An explicit value wins. Otherwise it is derived from the name, so two
        different ids differ and the same id is identical across runs and
        across processes.
        """
        if model in self.accuracies:
            return float(self.accuracies[model])
        digest = hashlib.sha256(("acc" + model).encode()).hexdigest()
        spread = int(digest[:8], 16) / 0xFFFFFFFF          # 0..1
        return float(min(0.95, max(0.05, self.accuracy - 0.15 + 0.3 * spread)))

    # -- completion ---------------------------------------------------------

    def complete(self, messages: List[Dict[str, str]], model: str,
                 temperature: float = 0.0, max_tokens: int = 1024,
                 sample: int = 0) -> Completion:
        self.calls += 1
        text = "".join(m.get("content", "") for m in messages)

        if STUB_MARKER in text and "GRADE" in text:
            reply = self._grade(text, sample)
            self.budget.record(len(text) // 4, 4, cached=False)
            return Completion(reply, model, len(text) // 4, 4, 0.0)

        answer = self._answer(model, text, sample)
        self.budget.record(len(text) // 4, 8, cached=False)
        return Completion(answer, model, len(text) // 4, 8, 0.0)

    def _answer(self, model: str, prompt: str, sample: int) -> str:
        digest = hashlib.sha256(
            ("%s|%s|%d|%d" % (model, prompt, sample, self.seed)).encode()
        ).hexdigest()[:12]
        return "%s(%s|%s)" % (STUB_MARKER, model, digest)

    def _grade(self, rubric_text: str, sample: int) -> str:
        model = self._model_from(rubric_text)
        threshold = self.true_accuracy(model)

        digest = hashlib.sha256(
            ("grade|%s|%d|%d" % (rubric_text, sample, self.seed)).encode()
        ).hexdigest()
        draw = int(digest[:8], 16) / 0xFFFFFFFF

        if self.judge_noise:
            # Shift the threshold per grading sample, so repeat gradings of one
            # answer can land on opposite sides of it.
            wobble = int(digest[8:16], 16) / 0xFFFFFFFF - 0.5
            threshold = min(1.0, max(0.0,
                                     threshold + 2 * self.judge_noise * wobble))

        return "GRADE: CORRECT" if draw < threshold else "GRADE: INCORRECT"

    @staticmethod
    def _model_from(text: str) -> str:
        start = text.find(STUB_MARKER + "(")
        if start < 0:
            return "unknown"
        start += len(STUB_MARKER) + 1
        end = text.find("|", start)
        return text[start:end] if end > start else "unknown"
