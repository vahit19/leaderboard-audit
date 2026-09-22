"""Run the CLI straight from a clone, with nothing installed.

    python la.py audit run_gsm8k_numeric/scores.csv

`pip install -e .` gives the same thing as a bare `la` command. This file
exists because the first thing a reader does is paste a command from the
README, and "command not found" at that moment costs more than the twenty
lines it takes to avoid.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from la.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
