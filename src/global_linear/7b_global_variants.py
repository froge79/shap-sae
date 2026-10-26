"""
Global-attribution variants for the linear probe, rescored against the saved
steering deltas (CPU, seconds — no new forward passes).

Responds to the objection that `Phi = mean signed SHAP` is not an importance
score (signed values cancel, so Phi tracks eval-vs-background mean shift) and
that a 100-row background makes that worse. Compares, on the filtered pool:

  signed_mean   Phi_j = mean_e phi_j(x_e)             (the published definition)
  mean_abs      mean_e |phi_j(x_e)|                   (the conventional global importance)
  signed_by_w   sign(w_j) * mean_e |phi_j(x_e)|       (mean_abs given a direction)

each against three backgrounds:

  bg100         the canonical 100-row train background (reproduces the published rows)
  bgfull        every train row (the interventional-SHAP limit; no sampling noise)
  boot<n>       B bootstrap backgrounds of n train rows, drawn with replacement;
                reports the distribution of rho over draws, not one draw

For a linear probe interventional SHAP is exactly phi_j(x) = w_j (x_j - mu_j) with
mu the background mean, so every variant is closed-form. The canonical case is
checked against the saved LinearExplainer output before anything is scored.

Causal effects are read from the existing `model_deltas_<tag>.json` files. They
do not depend on the background, so rescoring them is exact.

Output: 7b_global_variants/global_variants.{json,txt}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (  # noqa: E402
    CANONICAL_SEED,
    DEFAULT_N_EVAL,
    checkpoint_path,
    load_activations,
    load_eval_and_background,
    load_labels,
    output_path,
    sample_eval_rows,
)

LAYER = 7

# (label, directory under outputs, tag). Only runs whose candidate set does not
# depend on Phi: `--selection top` picks by |Phi| and would range-restrict.
RUNS = [
    ("all-token, all 114", "9_steer/steering_all_scaled", "steering_all_k114"),
    ("last-token, all 114", "9_steer/steering_all_scaled_lasttoken", "steering_all_k114"),
    ("random k=20 (scaled)", "9_steer/steering_random_k20_scaled", "steering_random_k20"),
    (
        "ablation random k=20",
        "9_steer/ablation_random_k20_scaled",
        "ablation_random_k20",
    ),
]


def load_deltas(rel_dir: str, tag: str):
    d = json.load(open(output_path(rel_dir, f"model_deltas_{tag}.json")))
    feats = np.asarray(sorted(int(k) for k in d["delta_abs"]), dtype=np.int64)
    dabs = np.asarray([d["delta_abs"][str(f)] for f in feats])
    dsig = np.asarray([d["delta_signed"][str(f)] for f in feats])
    if d["mode"] == "ablation":  # same orientation as 9_steer.orient_signed_effects
        dsig = -dsig
    return feats, dabs, dsig


def scores_from_background(w, x_eval, mu):
    """The three global scores, (n_pool,) each, for background mean `mu`."""
    phi = w * (x_eval - mu)  # (n_eval, n_pool), exact interventional SHAP
    mean_abs = np.abs(phi).mean(0)
    return {
        "signed_mean": phi.mean(0),
        "mean_abs": mean_abs,
        "signed_by_w": np.sign(w) * mean_abs,
    }


def rho(scores, dabs, dsig, kind):
    # Importance is |score| vs |delta|; directional is score vs signed delta.
    if kind == "importance":
        r, _ = spearmanr(np.abs(scores), dabs)
    else:
        r, _ = spearmanr(scores, dsig)
    return float(r)


def main(n_boot: int, boot_sizes: list[int], seed: int, out_dir: Path) -> None:
    probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))
    fi = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    w = np.asarray(probe.coef_[0], dtype=np.float64)[fi]

    x_eval, bg100, eval_idx, bg_idx = load_eval_and_background(
        layer=LAYER, n_eval=DEFAULT_N_EVAL
    )
    x_eval = x_eval[:, fi].astype(np.float64)
    bg100 = bg100[:, fi].astype(np.float64)

    train = load_activations("train", LAYER, "activations", mmap=True)
    n_train = train.shape[0]
    # Full train set through the shared sampler (a permutation prefix of length
    # n_train is every row), then slice columns before materialising.
    _, full_idx = sample_eval_rows(layer=LAYER, n_eval=DEFAULT_N_EVAL, n_background=n_train)
    x_train = np.asarray(train[full_idx][:, fi], dtype=np.float64)

    # Guard: the closed form must reproduce the saved LinearExplainer output.
    saved = np.load(output_path("7_shap_recompute", "shap_values_raw.npy"), mmap_mode="r")
    closed = w * (x_eval - bg100.mean(0))
    err = float(np.abs(closed - np.asarray(saved[:, fi])).max())
    assert err < 1e-5, f"closed form disagrees with saved SHAP (max err {err:.2e})"

    # Descriptives behind the objection.
    y_tr, y_sh = load_labels("train", LAYER), load_labels("shap", LAYER)
    desc = {
        "bg100_pos_frac": float(y_tr[bg_idx].mean()),
        "eval_pos_frac": float(y_sh[eval_idx].mean()),
        "train_pos_frac": float(y_tr.mean()),
        "closed_form_max_err": err,
    }
    mu = {"bg100": bg100.mean(0), "bgfull": x_train.mean(0)}
    for name, m in mu.items():
        phi_mean = w * (x_eval.mean(0) - m)
        desc[f"{name}_sign_agree_w"] = float(np.mean(np.sign(phi_mean) == np.sign(w)))
        desc[f"{name}_sum_phi"] = float(phi_mean.sum())
        desc[f"{name}_rho_absPhi_vs_absw"] = float(spearmanr(np.abs(phi_mean), np.abs(w))[0])
        desc[f"{name}_rho_absPhi_vs_absshift"] = float(
            spearmanr(np.abs(phi_mean), np.abs(x_eval.mean(0) - m))[0]
        )

    runs = {lab: load_deltas(d, t) for lab, d, t in RUNS}
    pos = {int(f): i for i, f in enumerate(fi)}

    def run_rhos(scores):
        out = {}
        for lab, (feats, dabs, dsig) in runs.items():
            s = scores[[pos[int(f)] for f in feats]]
            out[lab] = {k: rho(s, dabs, dsig, k) for k in ("importance", "directional")}
        return out

    result = {"descriptives": desc, "n_pool": int(len(fi)), "point": {}, "boot": {}}
    for bg_name, m in mu.items():
        for score_name, s in scores_from_background(w, x_eval, m).items():
            result["point"][f"{score_name}@{bg_name}"] = run_rhos(s)

    rng = np.random.default_rng(seed)
    for n in boot_sizes:
        draws: dict[str, dict] = {}
        for _ in range(n_boot):
            b = rng.integers(0, n_train, size=n)
            for score_name, s in scores_from_background(w, x_eval, x_train[b].mean(0)).items():
                for lab, r in run_rhos(s).items():
                    for kind, v in r.items():
                        draws.setdefault(f"{score_name}@boot{n}", {}).setdefault(
                            f"{lab}|{kind}", []
                        ).append(v)
        for key, d in draws.items():
            result["boot"][key] = {
                k: {
                    "mean": float(np.mean(v)),
                    "lo": float(np.percentile(v, 2.5)),
                    "hi": float(np.percentile(v, 97.5)),
                    "frac_pos": float(np.mean(np.asarray(v) > 0)),
                }
                for k, v in d.items()
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(out_dir / "global_variants.json", "w"), indent=2)
    text = render(result, n_boot, boot_sizes, n_train)
    (out_dir / "global_variants.txt").write_text(text)
    print(text)


def render(res, n_boot, boot_sizes, n_train) -> str:
    L = []
    d = res["descriptives"]
    L.append(f"Global-attribution variants — linear arm, pool n={res['n_pool']}, "
             f"bootstrap B={n_boot} (bg sizes {boot_sizes}, train n={n_train})")
    L.append("")
    L.append(f"class balance: bg100 {d['bg100_pos_frac']:.3f}  eval {d['eval_pos_frac']:.3f}  "
             f"train {d['train_pos_frac']:.3f}")
    L.append(f"closed-form vs saved LinearExplainer: max err {d['closed_form_max_err']:.1e}")
    for bg in ("bg100", "bgfull"):
        L.append(f"[{bg}] signed Phi: sign agrees with w for {d[f'{bg}_sign_agree_w']:.1%} of pool; "
                 f"sum Phi {d[f'{bg}_sum_phi']:+.3f}; "
                 f"rho(|Phi|,|w|)={d[f'{bg}_rho_absPhi_vs_absw']:.2f}, "
                 f"rho(|Phi|,|mean shift|)={d[f'{bg}_rho_absPhi_vs_absshift']:.2f}")
    for kind in ("importance", "directional"):
        L.append("")
        L.append(f"== {kind} faithfulness (Spearman rho vs causal effect) ==")
        for lab in res["point"]["signed_mean@bg100"]:
            L.append(f"-- {lab}")
            for key, r in res["point"].items():
                if kind == "directional" and key.startswith("mean_abs"):
                    continue  # unsigned score
                L.append(f"   {key:22s} {r[lab][kind]:+.3f}")
            for key, r in res["boot"].items():
                if kind == "directional" and key.startswith("mean_abs"):
                    continue
                b = r[f"{lab}|{kind}"]
                L.append(f"   {key:22s} {b['mean']:+.3f}  [{b['lo']:+.3f}, {b['hi']:+.3f}]  "
                         f"frac>0 {b['frac_pos']:.2f}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--boot-sizes", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    main(a.n_boot, a.boot_sizes, a.seed, a.out_dir or output_path("7b_global_variants"))
