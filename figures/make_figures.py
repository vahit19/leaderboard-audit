"""Build the two headline figures from the run tables.

Regenerate with:  python figures/make_figures.py

Each figure is written twice, light and dark. The dark version is stepped for
the dark surface rather than being an automatic inversion, so both are rendered
against the surface they will actually sit on -- a README is read in both
themes and a flipped light chart looks wrong in one of them.

Palette and mark rules follow the validated categorical set; the slot ordering
is the colour-blindness mechanism, so slots are used in order and never cycled.
Identity is never carried by colour alone: every series is directly labelled.
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")

SYSTEMS_ORDER = [
    "meta-llama/llama-3.3-70b-instruct",
    "openai/gpt-4o-mini",
    "mistralai/ministral-8b-2512",
    "openai/gpt-4.1-nano",
    "qwen/qwen-2.5-7b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

GRADERS = [
    ("ground truth", "run_gsm8k_numeric/results.jsonl"),
    ("gpt-4o-mini", "run_gsm8k_judge/results.jsonl"),
    ("gemini-2.5-flash-lite", "run_j_gemini-2.5-flash-lite/results.jsonl"),
    ("claude-3-haiku", "run_j_claude-3-haiku/results.jsonl"),
]

THEME = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e",
        "muted": "#898781", "grid": "#e1e0d9", "baseline": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "context": "#c3c2b7",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#898781", "grid": "#2c2c2a", "baseline": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "context": "#4a4a46",
    },
}

SHORT = {
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "mistralai/ministral-8b-2512": "ministral-8b",
    "openai/gpt-4.1-nano": "gpt-4.1-nano",
    "qwen/qwen-2.5-7b-instruct": "qwen-2.5-7b",
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b",
}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_scores(path: str):
    """Mean score per (system, item), majority-thresholded for repeat grades."""
    rows = collections.defaultdict(list)
    with open(os.path.join(ROOT, path), encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            rows[(r["system"], r["item"])].append(r["score"])
    return {k: float(np.mean(v)) for k, v in rows.items()}


def per_system_mean(scores, binarise=False):
    out = {}
    for system in SYSTEMS_ORDER:
        values = [v for k, v in scores.items() if k[0] == system]
        if binarise:
            values = [1.0 if v >= 0.5 else 0.0 for v in values]
        out[system] = float(np.mean(values)) if values else float("nan")
    return out


def build():
    truth = load_scores(GRADERS[0][1])
    means, ranks = {}, {}
    for name, path in GRADERS:
        scores = load_scores(path)
        m = per_system_mean(scores, binarise=(name != "ground truth"))
        means[name] = m
        order = sorted(SYSTEMS_ORDER, key=lambda s: -m[s])
        ranks[name] = {s: order.index(s) + 1 for s in SYSTEMS_ORDER}

    bias = {}
    truth_cells = truth
    for name, path in GRADERS[1:]:
        scores = load_scores(path)
        row = {}
        for system in SYSTEMS_ORDER:
            keys = [k for k in truth_cells if k[0] == system and k in scores]
            row[system] = float(np.mean(
                [(1.0 if scores[k] >= 0.5 else 0.0) - truth_cells[k]
                 for k in keys]))
        bias[name] = row
    return means, ranks, bias


# ---------------------------------------------------------------------------
# figure 1: rank under each grader
# ---------------------------------------------------------------------------

def figure_ranks(ranks, mode: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEME[mode]
    names = [g[0] for g in GRADERS]
    x = np.arange(len(names))

    # Emphasis form: the two systems that move carry the categorical slots in
    # order; every other system is context grey. Cycling hues across all six
    # would claim six identities matter when only two do.
    highlight = {
        "meta-llama/llama-3.3-70b-instruct": t["series"][0],
        "qwen/qwen-2.5-7b-instruct": t["series"][1],
    }

    fig, ax = plt.subplots(figsize=(9.2, 5.0), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])

    for system in SYSTEMS_ORDER:
        y = [ranks[n][system] for n in names]
        colour = highlight.get(system, t["context"])
        emphasised = system in highlight
        ax.plot(x, y, color=colour, linewidth=2.6 if emphasised else 1.6,
                zorder=3 if emphasised else 2,
                solid_capstyle="round", alpha=1.0 if emphasised else 0.85)
        # 2px surface ring keeps crossing markers readable where lines overlap
        ax.plot(x, y, "o", color=colour, markersize=9 if emphasised else 7,
                markeredgecolor=t["surface"], markeredgewidth=2,
                zorder=3 if emphasised else 2)
        # Direct labels: identity never rests on colour alone.
        ax.annotate(SHORT[system], (x[-1] + 0.08, y[-1]),
                    color=t["ink"] if emphasised else t["secondary"],
                    fontsize=9.5, va="center",
                    fontweight="bold" if emphasised else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(["graded by\n" + n if i else n
                        for i, n in enumerate(names)],
                       fontsize=10, color=t["secondary"])
    ax.set_yticks(range(1, len(SYSTEMS_ORDER) + 1))
    ax.set_yticklabels(["%d%s" % (i, {1: "st", 2: "nd", 3: "rd"}.get(i, "th"))
                        for i in range(1, len(SYSTEMS_ORDER) + 1)],
                       fontsize=10, color=t["muted"])
    ax.invert_yaxis()
    ax.set_xlim(-0.35, len(names) - 0.35 + 1.5)
    ax.set_ylim(len(SYSTEMS_ORDER) + 0.5, 0.5)

    ax.grid(axis="y", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])
    ax.tick_params(length=0)

    ax.set_title("Same 200 answers. Four graders. Four verdicts on who is best.",
                 color=t["ink"], fontsize=13, fontweight="bold",
                 loc="left", pad=16)
    ax.text(0, 1.015, "llama-3.3-70b is 1st on ground truth and 5th under "
                      "gpt-4o-mini — the answers never changed",
            transform=ax.transAxes, color=t["secondary"], fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT, "ranks-by-grader-%s.png" % mode)
    fig.savefig(path, dpi=200, facecolor=t["surface"])
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# figure 2: per-system bias, by judge
# ---------------------------------------------------------------------------

def figure_bias(bias, mode: str) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = THEME[mode]
    judges = [g[0] for g in GRADERS[1:]]
    order = sorted(SYSTEMS_ORDER,
                   key=lambda s: np.mean([bias[j][s] for j in judges]))

    fig, ax = plt.subplots(figsize=(9.2, 4.6), facecolor=t["surface"])
    ax.set_facecolor(t["surface"])

    y = np.arange(len(order))
    offsets = np.linspace(-0.22, 0.22, len(judges))

    # Zero is the reference the whole figure is read against, so it is the one
    # emphatic rule on the plot; the grid stays recessive behind it.
    ax.axvline(0, color=t["baseline"], linewidth=1.6, zorder=2)

    for j, judge in enumerate(judges):
        values = [100 * bias[judge][s] for s in order]
        ax.scatter(values, y + offsets[j], s=95, color=t["series"][j],
                   edgecolor=t["surface"], linewidth=2, zorder=4,
                   label=judge)

    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[s] for s in order], fontsize=10,
                       color=t["secondary"])
    ax.set_xlabel("points the grader adds to, or takes from, the true score",
                  fontsize=10, color=t["secondary"])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9.5, length=0)
    ax.tick_params(axis="y", length=0)

    ax.grid(axis="x", color=t["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["baseline"])

    span = max(abs(v) for row in bias.values() for v in row.values()) * 100
    ax.set_xlim(-span - 3.5, span + 3.5)
    # Direction hints sit at the foot of the zero line, clear of the subtitle.
    ax.set_ylim(-1.05, len(order) - 0.45)
    ax.text(-0.8, -0.85, "harsher", color=t["muted"], fontsize=9.5,
            ha="right", va="center", style="italic")
    ax.text(0.8, -0.85, "more generous", color=t["muted"], fontsize=9.5,
            ha="left", va="center", style="italic")

    legend = ax.legend(frameon=False, fontsize=9.5, ncol=3,
                       loc="upper center", bbox_to_anchor=(0.5, -0.24))
    for text in legend.get_texts():
        text.set_color(t["secondary"])

    ax.set_title("Each judge is wrong in its own shape",
                 color=t["ink"], fontsize=13, fontweight="bold",
                 loc="left", pad=30)
    ax.text(0, 1.045, "claude-3-haiku is generous to everyone, and most "
                      "generous to the weakest model",
            transform=ax.transAxes, color=t["secondary"], fontsize=10)

    fig.tight_layout()
    path = os.path.join(OUT, "grader-bias-%s.png" % mode)
    fig.savefig(path, dpi=200, facecolor=t["surface"], bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is needed to build the figures: pip install matplotlib",
              file=sys.stderr)
        return 2

    os.makedirs(OUT, exist_ok=True)
    means, ranks, bias = build()

    for mode in ("light", "dark"):
        print("wrote", os.path.relpath(figure_ranks(ranks, mode), ROOT))
        print("wrote", os.path.relpath(figure_bias(bias, mode), ROOT))

    # The numbers behind the figures, so a reader can check them without
    # re-running anything.
    table = {
        "rank_by_grader": {g: {SHORT[s]: ranks[g][s] for s in SYSTEMS_ORDER}
                           for g in ranks},
        "score_by_grader": {g: {SHORT[s]: round(means[g][s], 4)
                                for s in SYSTEMS_ORDER} for g in means},
        "bias_by_judge": {j: {SHORT[s]: round(bias[j][s], 4)
                              for s in SYSTEMS_ORDER} for j in bias},
    }
    with open(os.path.join(OUT, "figure_data.json"), "w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2, sort_keys=True)
    print("wrote figures/figure_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
