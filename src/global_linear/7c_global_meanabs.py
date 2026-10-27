"""
Global SHAP importance as mean |phi| against the FULL train set (linear probe).

`7_shap_recompute` keeps Phi = mean *signed* SHAP, which is the right object for a
directional claim but not a global importance score: signed values cancel across
examples, so Phi mostly measures how the eval rows differ from a 100-row
background. This computes the conventional score

    A_j = mean_e |phi_j(x_e)|        over the canonical 1000 eval rows,

against every train row as background (6,696 rows; the interventional-SHAP limit,
so there is no background sampling noise). For the linear probe
phi_j(x) = w_j (x_j - mu_j), mu the background mean, so this is closed form; the
canonical 100-row case is checked against the saved LinearExplainer output first.

Output (filtered 114-feature axis, aligned with 3_shap/shap_feature_indices.npy):
  7c_global_meanabs/meanabs_linear.npy      (114,)
  7c_global_meanabs/meanabs_linear_full.npy (32768,)  every feature, for ranking tables
  7c_global_meanabs/feature_indices.npy     (114,)
  7c_global_meanabs/meta_linear.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (  # noqa: E402
    DEFAULT_N_EVAL,
    checkpoint_path,
    load_activations,
    load_eval_and_background,
    output_path,
    sample_eval_rows,
)

LAYER = 7


def main() -> None:
    probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))
    fi = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    w = np.asarray(probe.coef_[0], dtype=np.float64)[fi]

    x_eval, bg100, eval_idx, _ = load_eval_and_background(layer=LAYER, n_eval=DEFAULT_N_EVAL)
    x_eval = x_eval[:, fi].astype(np.float64)

    saved = np.load(output_path("7_shap_recompute", "shap_values_raw.npy"), mmap_mode="r")
    err = float(np.abs(w * (x_eval - bg100[:, fi].mean(0)) - np.asarray(saved[:, fi])).max())
    assert err < 1e-5, f"closed form disagrees with LinearExplainer (max err {err:.2e})"

    train = load_activations("train", LAYER, "activations", mmap=True)
    n_train = train.shape[0]
    _, bg_idx = sample_eval_rows(layer=LAYER, n_eval=DEFAULT_N_EVAL, n_background=n_train)
    mu = np.asarray(train[bg_idx][:, fi], dtype=np.float64).mean(0)

    score = np.abs(w * (x_eval - mu)).mean(0)
    # Same score on every feature (ranking-agreement tables use all non-zero weights).
    w_all = np.asarray(probe.coef_[0], dtype=np.float64)
    ev_all = np.asarray(load_activations("shap", LAYER, "activations", mmap=True)[eval_idx])
    mu_all = np.asarray(train[bg_idx], dtype=np.float64).mean(0)
    full = np.abs(w_all * (ev_all - mu_all)).mean(0)
    assert np.allclose(full[fi], score)

    out = output_path("7c_global_meanabs")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "meanabs_linear.npy", score)
    np.save(out / "meanabs_linear_full.npy", full)
    np.save(out / "feature_indices.npy", fi)
    json.dump(
        {"n_eval": len(eval_idx), "n_background": int(n_train), "closed_form_max_err": err},
        open(out / "meta_linear.json", "w"),
        indent=2,
    )
    print(f"wrote {out}/meanabs_linear.npy  shape={score.shape}  bg={n_train}  check err={err:.1e}")


if __name__ == "__main__":
    main()
