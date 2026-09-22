"""Reading API keys from a .env file without a dependency.

Keys are loaded into the process environment and nowhere else. They are never
written to results, the cache, the manifest or a log line, and this module has
no function that returns a key's value -- `loaded_keys` reports names only, so
printing a diagnostic cannot leak one.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

__all__ = ["load_dotenv", "loaded_keys", "require_key"]

#: Names checked for a key, in order. Both cases appear in the wild.
OPENROUTER_NAMES = ("OPENROUTER_API_KEY", "openrouter_api_key")


def load_dotenv(path: Optional[str] = None, override: bool = False) -> List[str]:
    """Load KEY=value lines from a .env file. Returns the names set.

    Walks up from the working directory if no path is given, so the file can
    live at the repository root while commands run from a subdirectory.
    """
    if path is None:
        path = _find_dotenv()
    if path is None or not os.path.exists(path):
        return []

    names: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name = name.strip()
            value = value.strip().strip('"').strip("'")
            if not name:
                continue
            if override or name not in os.environ:
                os.environ[name] = value
            names.append(name)
    return names


def _find_dotenv(start: Optional[str] = None, depth: int = 4) -> Optional[str]:
    here = os.path.abspath(start or os.getcwd())
    for _ in range(depth):
        candidate = os.path.join(here, ".env")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def loaded_keys() -> Dict[str, bool]:
    """Which known keys are present. Names and presence only, never values."""
    return {
        "openrouter": any(os.environ.get(n) for n in OPENROUTER_NAMES),
        "huggingface": bool(os.environ.get("HUGGINGFACE_HUB_TOKEN")),
    }


def require_key(names=OPENROUTER_NAMES) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(
        "no API key found. Set %s in the environment or in a .env file at the "
        "repository root." % names[0]
    )
