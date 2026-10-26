"""
Step 27c — aggregate the minimal-pair seed sweep (seeds x edit x distractor mode).

Reads run3/outputs/24_minimal_pairs/seeds/s<seed>_<random|active>/summary_*.json
(written by 24_minimal_pairs.py --seed S --out-dir ...) and reports, per
(edit, distractors, method), the mean / std / min / max across seeds of the
per-seed mean AUROC and causal-recovery rho, plus the fraction of seeds above
chance. Descriptive only: seeds redraw pairs from one held-out pool, so they
overlap and are not independent replicates.

Also reports the position contrasts the reviewer's critique turns on, as
same-seed differences (pools differ, so not strictly paired): insert - insert-interior
(position, type held fixed) and remove - insert-interior (type, position fixed).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from utils import output_path

EDITS = ("remove", "insert", "insert-interior", "substitute")
METHODS = ("local_SHAP", "global_SHAP", "Probe weights")
root = output_path("24_minimal_pairs", "seeds")

data: dict[tuple[str, str], dict[int, dict]] = {}
for d in sorted(root.glob("s*_*")):
    seed_s, mode = d.name[1:].split("_")
    suffix = "_actdist" if mode == "active" else ""
    for e in EDITS:
        f = d / f"summary_{e}_n100{suffix}.json"
        if f.exists():
            data.setdefault((e, mode), {})[int(seed_s)] = json.load(open(f))

lines = ["Minimal-pair seed sweep (descriptive; seeds overlap)", "=" * 78]
out = {}
for mode in ("random", "active"):
    lines.append(f"\n### distractors = {mode}")
    for e in EDITS:
        runs = data.get((e, mode), {})
        if not runs:
            continue
        seeds = sorted(runs)
        sa = np.array([runs[s]["readout_sign_accuracy"] for s in seeds])
        lines.append(f"\n{e}  (n_seeds={len(seeds)})  sign acc mean {sa.mean():.3f} "
                     f"[{sa.min():.2f},{sa.max():.2f}]")
        lines.append(f"  {'method':14s} {'AUROC mean±sd':>16s} {'min':>6s} {'max':>6s} {'>0.5':>5s}"
                     f" | {'rho mean±sd':>16s} {'>0':>4s}")
        for m in METHODS:
            a = np.array([runs[s]["methods"][m]["mean_auroc_moved"] for s in seeds])
            r = np.array([runs[s]["methods"][m]["mean_rho_recovery"] for s in seeds])
            out[f"{e}|{mode}|{m}"] = {"auroc": a.tolist(), "rho": r.tolist(), "seeds": seeds}
            lines.append(
                f"  {m:14s} {a.mean():7.3f}±{a.std(ddof=1) if len(a) > 1 else 0:.3f} "
                f"{a.min():6.3f} {a.max():6.3f} {int((a > 0.5).sum()):2d}/{len(a):<2d}"
                f" | {r.mean():7.3f}±{r.std(ddof=1) if len(r) > 1 else 0:.3f} "
                f"{int((r > 0).sum()):2d}/{len(r)}"
            )

lines.append("\n### Same-seed contrasts (sentence pools differ, so not strictly paired), local_SHAP AUROC, distractors=random")
for a_, b_ in (("insert", "insert-interior"), ("remove", "insert-interior"), ("insert", "remove")):
    ra, rb = data.get((a_, "random"), {}), data.get((b_, "random"), {})
    common = sorted(set(ra) & set(rb))
    if not common:
        continue
    diff = np.array([ra[s]["methods"]["local_SHAP"]["mean_auroc_moved"]
                     - rb[s]["methods"]["local_SHAP"]["mean_auroc_moved"] for s in common])
    lines.append(f"  {a_} - {b_}: mean {diff.mean():+.3f}, sd {diff.std(ddof=1):.3f}, "
                 f"positive in {int((diff > 0).sum())}/{len(diff)} seeds")
    out[f"contrast|{a_}-{b_}"] = diff.tolist()

text = "\n".join(lines)
print(text)
(root.parent / "seed_sweep_summary.txt").write_text(text)
(root.parent / "seed_sweep_summary.json").write_text(json.dumps(out, indent=2))
