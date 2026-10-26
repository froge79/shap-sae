"""
Plot local-vs-global per-example faithfulness from outputs/9_local_steer/ and
outputs/14_local_mlp_steer/ (summary.json + paired_tests.json).

The headline claim in this section is a *reversal pattern* across four
conditions (block A/B x importance/directional) and two arms: local SHAP
wins decisively pre-activity-control, the importance-ranking win does not
survive the control (and flips in the MLP arm), while the directional win
survives in both arms. A grouped bar chart with 95% CI whiskers makes that
crossover visible at a glance; the 8-row table it replaces in the manuscript
requires reading six numeric columns per row to reconstruct the same pattern.

Usage
-----
  uv run python src/local/plot_local_vs_global.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import output_path

LINEAR_DIR = output_path("9_local_steer")
MLP_DIR = output_path("14_local_mlp_steer")
OUT_DIR = output_path("9_local_steer")

# (summary.json key prefix, paired_tests block key, display label)
BLOCKS = [
    ("", "A:importance", "A: importance"),
    ("", "A:directional", "A: directional"),
    ("active:", "B:importance", "B: importance"),
    ("active:", "B:directional", "B: directional"),
]


def load_arm(summary_dir: Path) -> dict:
    with open(summary_dir / "summary.json") as f:
        summary = json.load(f)
    with open(summary_dir / "paired_tests.json") as f:
        paired = json.load(f)
    return {"summary": summary, "paired": paired}


def block_stats(arm: dict) -> list[dict]:
    """
    One row per BLOCKS entry: local/global mean rho with crossed 95% CIs and the
    crossed p for the local-global gap.

    Everything comes from the crossed (examples x features) bootstrap in
    analyze_local_vs_global.py. The examples are scored on largely the same
    features, so std/sqrt(n) error bars and the examples-only Wilcoxon p would
    treat correlated observations as independent; they are deliberately not used.
    """
    paired = arm["paired"]
    rows = []
    for _prefix, block_key, label in BLOCKS:
        cb = paired["blocks"][block_key].get("crossed")
        if cb is None:
            raise RuntimeError(
                f"{block_key}: no crossed bootstrap in paired_tests.json - rerun "
                "the local scripts and analyze_local_vs_global.py first"
            )
        rows.append(
            {
                "label": label,
                "local_mean": cb["point_local"],
                "local_ci": cb["local_ci95"],
                "global_mean": cb["point_global"],
                "global_ci": cb["global_ci95"],
                "p": cb["p_two_sided"],
            }
        )
    return rows


def _yerr(means: list[float], cis: list[list[float]]) -> np.ndarray:
    """Asymmetric percentile CIs -> matplotlib's (2, n) yerr."""
    return np.array(
        [[m - lo for m, (lo, _) in zip(means, cis)], [hi - m for m, (_, hi) in zip(means, cis)]]
    )


def sig_marker(p: float) -> str:
    # The crossed bootstrap (B=1000) cannot resolve p below ~2e-3, so no "***".
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def y_extent(rows: list[dict]) -> tuple[float, float]:
    """Lowest and highest CI bound across both bars of every block."""
    lo = min(min(r["local_ci"][0], r["global_ci"][0]) for r in rows)
    hi = max(max(r["local_ci"][1], r["global_ci"][1]) for r in rows)
    return lo, hi


def plot_arm(
    rows: list[dict], title: str, ax: plt.Axes, y_bot: float, y_top: float
) -> None:
    """y_bot / y_top span both arms: the panels share a y-axis."""
    x = np.arange(len(rows))
    width = 0.35

    local_means = [r["local_mean"] for r in rows]
    local_cis = [r["local_ci"] for r in rows]
    global_means = [r["global_mean"] for r in rows]
    global_cis = [r["global_ci"] for r in rows]

    ax.bar(
        x - width / 2, local_means, width, yerr=_yerr(local_means, local_cis),
        capsize=3, label="Local SHAP", color="#4C78A8",
    )
    ax.bar(
        x + width / 2, global_means, width, yerr=_yerr(global_means, global_cis),
        capsize=3, label="Global $\\Phi$", color="#F58518",
    )

    ax.axhline(0.0, color="0.3", linewidth=0.9, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_ylabel("Mean Spearman $\\rho$ (crossed 95% CI)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)

    span = y_top - y_bot
    for xi, r in enumerate(rows):
        top = max(r["local_ci"][1], r["global_ci"][1])
        ax.text(
            xi, top + 0.03 * span, sig_marker(r["p"]),
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_ylim(y_bot - 0.12 * span, y_top + 0.18 * span)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--linear-dir", type=Path, default=LINEAR_DIR)
    p.add_argument("--mlp-dir", type=Path, default=MLP_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    linear_rows = block_stats(load_arm(args.linear_dir))
    mlp_rows = block_stats(load_arm(args.mlp_dir))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    extents = [y_extent(linear_rows), y_extent(mlp_rows)]
    y_bot, y_top = min(e[0] for e in extents), max(e[1] for e in extents)
    plot_arm(linear_rows, "Linear arm", axes[0], y_bot, y_top)
    plot_arm(mlp_rows, "MLP arm", axes[1], y_bot, y_top)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Local vs. global per-example faithfulness ($n$=100 sentences, ablation)\n"
        "CIs and gap significance from a crossed sentences $\\times$ features bootstrap",
        fontsize=12,
    )
    fig.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "local_vs_global_faithfulness.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
