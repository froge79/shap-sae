"""
Give IG and GA the same aggregation as global SHAP: mean |attribution| over the
canonical 1000 eval rows (CPU, no LM forward passes).

`7d_rescore_global_meanabs.py` switched SHAP importance to mean |phi| but left IG/GA
as |signed mean|. That confounds method with aggregation: in the linear arm IG = GA =
SHAP per example, so the signed-mean vs mean-|.| gap is entirely aggregation; in the
MLP arm it is partly aggregation. This script computes, per arm and per baseline
(100-row = the existing IG/GA baseline; full train = SHAP's background):

    A_j^IG = mean_e |IG_j(x_e)|,   A_j^GA = mean_e |GA_j(x_e)|

then rescored against the saved `--selection all` steering effects (importance only;
direction is unchanged), with BH within run exactly as 7d does.

Output: 7c_global_meanabs/meanabs_{ig,ga}_{mlp,linear}_{bg100,full}.npy (114,)
        7f_meanabs_ig_ga/{summary.json,summary.txt}
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
from utils import (  # noqa: E402
    DEFAULT_N_EVAL,
    benjamini_hochberg,
    checkpoint_path,
    load_activations,
    load_eval_and_background,
    output_path,
    sample_eval_rows,
)

LAYER = 7
RUNS = {  # (arm, scope) -> run dir
    ("linear", "all-token"): "9_steer/steering_all_scaled",
    ("linear", "last-token"): "9_steer/steering_all_scaled_lasttoken",
    ("mlp", "all-token"): "14_mlp_steer/steering_all_scaled",
    ("mlp", "last-token"): "14_mlp_steer/steering_all_scaled_lasttoken",
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mlp_per_example(probe, x, baseline, grad_fn, n_steps=50):
    """Per-example IG (trapezoid) and GA, each (n, F); same definitions as 13_mlp_ranking."""
    alphas = np.linspace(0, 1, n_steps)
    ig, ga = np.empty_like(x), np.empty_like(x)
    for i, xi in enumerate(x):
        grads = grad_fn(probe, baseline + alphas[:, None] * (xi - baseline))
        ig[i] = (xi - baseline) * np.trapezoid(grads, alphas, axis=0)
        ga[i] = (xi - baseline) * grad_fn(probe, xi)
    return ig, ga


def score(imp: np.ndarray, effects: dict) -> tuple[float, float]:
    f = list(effects)
    r, p = spearmanr([abs(imp[int(k)]) for k in f], [abs(effects[k]) for k in f])
    return float(r), float(p)


def main() -> None:
    rk = _load("mlp_rank13", _SRC / "global_mlp" / "13_mlp_ranking.py")
    out = output_path("7c_global_meanabs")

    x_eval, bg100, _, _ = load_eval_and_background(layer=LAYER, n_eval=DEFAULT_N_EVAL)
    train = load_activations("train", LAYER, "activations", mmap=True)
    _, order = sample_eval_rows(layer=LAYER, n_eval=DEFAULT_N_EVAL, n_background=train.shape[0])

    # ---- MLP arm ------------------------------------------------------------
    payload = joblib.load(checkpoint_path(f"mlp_probe_layer_{LAYER}.joblib"))
    probe, fi = payload["probe"], np.asarray(payload["feature_indices"])
    xm = np.asarray(x_eval[:, fi], dtype=np.float32)
    mu = {
        "bg100": np.asarray(bg100[:, fi], dtype=np.float32).mean(0),
        "full": np.asarray(train[order][:, fi], dtype=np.float32).mean(0),
    }
    scores: dict[tuple[str, str, str], np.ndarray] = {}
    for bg, base in mu.items():
        ig, ga = mlp_per_example(probe, xm, base, rk.mlp_logit_grad)
        # sanity: signed mean must reproduce the saved (100-row) IG/GA
        if bg == "bg100":
            for nm, arr in (("ig", ig), ("ga", ga)):
                saved = np.load(output_path("13_mlp_ranking", f"{nm}_scores.npy"))
                assert np.abs(arr.mean(0) - saved).max() < 1e-4, f"{nm} signed mean != saved"
        for nm, arr in (("ig", ig), ("ga", ga)):
            scores[("mlp", nm, bg)] = np.abs(arr).mean(0)

    # ---- linear arm: IG = GA = w_j (x_j - mu_j) per example ------------------
    ckpt = Path(os.environ.get("SHAP_SAE_CHECKPOINTS", "checkpoints"))
    w = np.asarray(joblib.load(ckpt / "probe_layer_7.joblib").coef_[0], dtype=np.float64)
    fl = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    assert np.array_equal(fl, fi)
    xl = np.asarray(x_eval[:, fl], dtype=np.float64)
    for bg, base in (("bg100", np.asarray(bg100[:, fl], np.float64).mean(0)),
                     ("full", np.asarray(train[order][:, fl], np.float64).mean(0))):
        a = np.abs(w[fl]) * np.abs(xl - base).mean(0)
        scores[("linear", "ig", bg)] = scores[("linear", "ga", bg)] = a
    ref = np.load(out / "meanabs_linear.npy")
    assert np.abs(scores[("linear", "ig", "full")] - ref).max() < 1e-6, "linear != 7c meanabs_linear"

    for (arm, nm, bg), v in scores.items():
        np.save(out / f"meanabs_{nm}_{arm}_{bg}.npy", v)

    # ---- rescore importance vs saved `all` effects --------------------------
    res: dict = {}
    lines = ["Importance rho (Spearman |score| vs |delta|), k=114; IG/GA aggregated as mean |.|",
             f"{'arm, scope':<22}{'method':<20}{'rho':>8}{'p':>10}"]
    for (arm, scope), rel in RUNS.items():
        d = json.load(open(next(output_path(rel).glob("model_deltas_*.json"))))["delta_abs"]
        fi_arm = fl if arm == "linear" else fi
        full = lambda v: np.bincount(fi_arm, weights=v, minlength=32768)  # noqa: E731
        shap = full(np.load(out / f"meanabs_{arm}.npy"))
        rows = {"SHAP mean|phi| (full)": shap}
        for nm in ("ig", "ga"):
            for bg in ("bg100", "full"):
                rows[f"{nm.upper()} mean|.| ({bg})"] = full(scores[(arm, nm, bg)])
        stats = {k: dict(zip(("rho", "p"), score(v, d))) for k, v in rows.items()}
        # BH family: SHAP + IG/GA (full bg) importance; linear IG/GA identical -> one test
        fam_keys = ["SHAP mean|phi| (full)", "IG mean|.| (full)", "GA mean|.| (full)"]
        if arm == "linear":
            fam_keys = fam_keys[:2]  # IG == GA == SHAP per example: all three are one test
        bh = benjamini_hochberg({k: stats[k]["p"] for k in fam_keys})
        for k, v in stats.items():
            v.update(q_bh_within_methods=bh.get(k, {}).get("q"))
            lines.append(f"{arm + ', ' + scope:<22}{k:<20}{v['rho']:>+8.3f}{v['p']:>10.4g}")
        res[f"{arm}|{scope}"] = stats
    d = output_path("7f_meanabs_ig_ga")
    d.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(d / "summary.json", "w"), indent=2)
    (d / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
