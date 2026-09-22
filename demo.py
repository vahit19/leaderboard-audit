"""Generate the example table and audit it, in one command, anywhere.

    python demo.py

The Makefile does the same thing, but make is not present on a default Windows
install and a reader who has to fix their toolchain before seeing a number
usually does not see the number.
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
EXAMPLE = os.path.join(ROOT, "examples", "example.csv")


def run(args, cwd=None, env=None):
    print("$ " + " ".join(args))
    result = subprocess.run(args, cwd=cwd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> int:
    if not os.path.exists(EXAMPLE):
        run([sys.executable, os.path.join("examples", "make_example.py"),
             "--out", EXAMPLE], cwd=ROOT)
        print()

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    run([sys.executable, "-m", "la.cli", EXAMPLE,
         "--out", os.path.join(ROOT, "results"), "--top", "6"],
        cwd=ROOT, env=env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
