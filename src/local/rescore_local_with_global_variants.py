"""
Rescore the saved local-vs-global runs with alternative global baselines (CPU).

Swaps `global_phi_filtered.npy` in a scratch copy of each local run and re-runs
`analyze_local_vs_global.analyze_dir`, so the numbers come from the same code
path (and the same crossed bootstrap) as the published tables.

CAVEAT — read before quoting. Each example's candidate set is the union of its
top-k local features and the top-k *original* |Phi| features, and the Deltas were
measured only on that union. Rescoring changes the score, not the candidates:
a feature that the new global score would rank in its top-k but the old one did
not was never intervened on. This is therefore a rescoring on the original
candidate sets, not a rerun of the experiment with the better baseline.

Variants (linear arm, from the closed form w_j (x_j - mu_j)):
  signed_bg100    published Phi
  signed_bgfull   Phi against every train row — isolates background size
  absmean_bg100   sign(w) * mean|phi|, 100-row background
  absmean_bgfull  sign(w) * mean|phi|, every train row
The sign(w) factor lets one file serve both kinds: |score| is exactly mean|phi|
(importance) and the sign is the probe's own direction (directional).

MLP arm: mean|phi| from the saved DeepSHAP values (bg100, importance only).

Output: 7b_global_variants/local_rescored/<source>__<variant>/paired_tests.{json,txt}
and 7b_global_variants/local_rescored/summary.txt
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
from utils import (  # noqa: E402
    DEFAULT_N_EVAL,
    checkpoint_path,
    load_activations,
    load_eval_and_background,
    output_path,
    sample_eval_rows,
)

LAYER = 7


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ana = _load("analyze_lvg", _SRC / "local" / "analyze_local_vs_global.py")


def linear_variants() -> dict[str, np.ndarray]:
    probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))
    fi = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    w = np.asarray(probe.coef_[0], dtype=np.float64)[fi]
    x_eval, bg100, _, _ = load_eval_and_background(layer=LAYER, n_eval=DEFAULT_N_EVAL)
    x_eval = x_eval[:, fi].astype(np.float64)
    train = load_activations("train", LAYER, "activations", mmap=True)
    _, order = sample_eval_rows(
        layer=LAYER, n_eval=DEFAULT_N_EVAL, n_background=train.shape[0]
    )
    mu = {
        "bg100": bg100[:, fi].astype(np.float64).mean(0),
        "bgfull": np.asarray(train[order][:, fi], dtype=np.float64).mean(0),
    }
    out = {}
    for bg, m in mu.items():
        phi = w * (x_eval - m)
        out[f"signed_{bg}"] = phi.mean(0)
        out[f"absmean_{bg}"] = np.sign(w) * np.abs(phi).mean(0)
    return out


def mlp_variants() -> dict[str, np.ndarray]:
    sv = np.load(output_path("12_mlp_shap", "shap_values_raw.npy"))
    return {"signed_bg100": sv.mean(0), "absmean_bg100": np.abs(sv).mean(0)}


def main() -> None:
    root = output_path("7b_global_variants", "local_rescored")
    jobs = {
        "9_local_steer": linear_variants(),
        "local_shap_faithfulness": linear_variants(),
        "14_local_mlp_steer": mlp_variants(),
    }
    lines = ["Local vs global — rescored on the ORIGINAL candidate sets (see script docstring)", ""]
    for src, variants in jobs.items():
        for name, score in variants.items():
            dst = root / f"{src}__{name}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(output_path(src), dst, ignore=shutil.ignore_patterns("*.png"))
            old = np.load(dst / "global_phi_filtered.npy")
            assert old.shape == score.shape
            np.save(dst / "global_phi_filtered.npy", score)
            res = _ana.analyze_dir(dst, "ablation")
            (dst / "paired_tests.json").write_text(json.dumps(res, indent=2))
            (dst / "paired_tests.txt").write_text(_ana.render(res))
            lines.append(f"[{src}  global={name}]")
            for blk, r in res["blocks"].items():
                c = r["crossed"]
                lines.append(
                    f"  {blk:16s} local {r['mean_local']:+.3f}  global {r['mean_global']:+.3f}  "
                    f"gap {r['mean_gap']:+.3f}  crossed CI [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]  "
                    f"p={c['p_two_sided']:.3f}"
                )
            lines.append("")
    text = "\n".join(lines)
    (root / "summary.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
