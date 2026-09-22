"""Content-addressed cache for model calls.

Every completion is keyed by a hash of everything that determines it: model,
messages, temperature, max tokens, and a sample index. Two consequences, both
required for this to be usable on real work:

  Nothing is paid for twice. A run that dies at item 400 of 600 resumes by
  replaying 400 cache hits, and a re-analysis costs nothing at all.

  A run can be reproduced offline. With the cache alongside the results,
  anyone can re-execute the pipeline through `ReplayProvider` and get the same
  numbers without an API key.

The sample index is part of the key on purpose. Repeat runs at a non-zero
temperature are supposed to differ -- that variation is the measurement this
whole system exists to quantify -- so sample 2 must not be served sample 1's
answer.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from typing import Any, Dict, Optional

__all__ = ["CallKey", "DiskCache"]


class CallKey:
    """Everything that determines a completion, hashed into one id."""

    __slots__ = ("model", "messages", "temperature", "max_tokens", "sample",
                 "extra")

    def __init__(
        self,
        model: str,
        messages: Any,
        temperature: float,
        max_tokens: int,
        sample: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.messages = messages
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.sample = int(sample)
        self.extra = extra or {}

    def digest(self) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": self.messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "sample": self.sample,
                "extra": self.extra,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def __repr__(self) -> str:
        return "CallKey(%s, sample=%d, %s)" % (
            self.model, self.sample, self.digest()[:12]
        )


class DiskCache:
    """Sharded JSON files on disk, safe across threads.

    Sharded by the first two hex characters so a long run does not put tens of
    thousands of files in one directory, which is slow to list on every
    filesystem people actually use.
    """

    def __init__(self, root: str, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if enabled:
            os.makedirs(root, exist_ok=True)

    def _path(self, digest: str) -> str:
        return os.path.join(self.root, digest[:2], digest + ".json")

    def get(self, key: CallKey) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        path = self._path(key.digest())
        if not os.path.exists(path):
            with self._lock:
                self.misses += 1
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                value = json.load(fh)
        except (OSError, ValueError):
            # A truncated file from a killed process is a miss, not a crash.
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return value

    def put(self, key: CallKey, value: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key.digest())
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp.%d" % threading.get_ident()
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(value, fh, ensure_ascii=False)
        os.replace(tmp, path)          # atomic, so a kill never leaves a half file
        with self._lock:
            self.writes += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses,
                    "writes": self.writes}
