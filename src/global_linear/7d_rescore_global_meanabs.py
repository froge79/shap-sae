"""
Rescore the saved global-faithfulness runs with mean |phi| as the SHAP importance
score (CPU, no new forward passes).

The global SHAP *importance* score becomes A_j = mean_e |phi_j(x_e)| against the
full train set (`7c_global_meanabs`, `12c_mlp_meanabs`). The *directional* score
stays the signed mean Phi: a sign cannot come from an unsigned score, and the
sign test of Phi is its own claim. IG, GA, probe weights/saliency are untouched.

Steering effects do not depend on any attribution when the candidate set is
`random` or `all`, so the saved `model_deltas_<tag>.json` are exactly what a rerun
would measure. `top*` runs choose candidates by |Phi| and are NOT rescored.

Reads  $SHAP_SAE_OUTPUTS (the lineage being rescored) and writes into
$SHAP_SAE_OUT_NEW (default: <outputs>_meanabs): per-run `faithfulness_<tag>.txt`
and `_bh.json` in the original layout (so `9d_bh_family_sensitivity.py` and
`paper/check_bh_numbers.py` run unchanged against the new root), the seed sweeps,
and the ranking-agreement tables.

    SHAP_SAE_OUTPUTS=run3/outputs SHAP_SAE_OUT_NEW=run3/outputs_meanabs \
      uv run python src/global_linear/7d_rescore_global_meanabs.py
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
from utils import benjamini_hochberg, output_path  # noqa: E402

OUT_NEW = Path(os.environ.get("SHAP_SAE_OUT_NEW", str(output_path()) + "_meanabs"))
ALPHA = 0.05

RUN_DIRS = {  # arm -> run directories rescored (candidate set independent of Phi)
    "linear": [
        "9_steer/steering_all_scaled",
        "9_steer/steering_all_scaled_lasttoken",
        "9_steer/steering_random_k20_scaled",
        "9_steer/steering_random_k20_constant",
        "9_steer/ablation_random_k20_scaled",
    ],
    "mlp": [
        "14_mlp_steer/steering_all_scaled",
        "14_mlp_steer/steering_all_scaled_lasttoken",
        "14_mlp_steer/steering_random_k20_scaled",
        "14_mlp_steer/steering_random_k20_constant",
        "14_mlp_steer/ablation_random_k20_scaled",
    ],
}
SEED_SWEEPS = {
    "linear": ("9_seed_sweep", "9_steer/steering_all_scaled"),
    "mlp": ("14_mlp_seed_sweep", "14_mlp_steer/steering_all_scaled"),
}
UNSIGNED = {"linear": set(), "mlp": {"Probe saliency"}}
NOTE = (
    "NOTE: SHAP/IG/GA importance = mean |attribution| (SHAP and IG/GA baseline: full train set, "
    "6696 rows); directional = signed mean (100-row background). Probe scores unchanged.\n"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_seed9a = _load("seed9a", _SRC / "global_linear" / "9a_seed_sweep.py")


def expand(vec, fi, n=32768):
    full = np.zeros(n)
    full[fi] = vec
    return full


def load_methods(arm: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """name -> (importance scores, directional scores), all (32768,) full width."""
    meanabs_pool = output_path("7c_global_meanabs")
    if arm == "linear":
        fi = np.load(output_path("3_shap", "shap_feature_indices.npy"))
        probe = joblib.load(
            Path(os.environ.get("SHAP_SAE_CHECKPOINTS", "checkpoints")) / "probe_layer_7.joblib"
        )
        w = np.asarray(probe.coef_[0], dtype=np.float64)
        ig = np.load(output_path("8_rankings_recompute", "ig_scores.npy")).astype(np.float64)
        ga = np.load(output_path("8_rankings_recompute", "ga_scores.npy")).astype(np.float64)
        phi = np.load(output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")).astype(
            np.float64
        )
        meanabs = expand(np.load(meanabs_pool / "meanabs_linear.npy"), fi)
        # IG/GA importance gets the same aggregation as SHAP (mean |.|, full-train
        # baseline; 7f_meanabs_ig_ga.py). Direction stays the signed mean.
        ig_imp = expand(np.load(meanabs_pool / "meanabs_ig_linear_full.npy"), fi)
        ga_imp = expand(np.load(meanabs_pool / "meanabs_ga_linear_full.npy"), fi)
        return {
            "SHAP": (meanabs, phi),
            "Probe weights": (w, w),
            "IG": (ig_imp, ig),
            "GA": (ga_imp, ga),
        }
    rk = output_path("13_mlp_ranking")
    fi = np.load(rk / "feature_indices.npy")
    sig = np.load(rk / "shap_scores_signed.npy") if (rk / "shap_scores_signed.npy").exists() else np.load(rk / "shap_scores.npy")
    ex = lambda n: expand(np.load(rk / n).astype(np.float64), fi)  # noqa: E731
    meanabs = expand(np.load(meanabs_pool / "meanabs_mlp.npy"), fi)
    return {
        "SHAP": (meanabs, expand(sig.astype(np.float64), fi)),
        "Probe saliency": (ex("probe_scores.npy"),) * 2,
        "IG": (expand(np.load(meanabs_pool / "meanabs_ig_mlp_full.npy"), fi), ex("ig_scores.npy")),
        "GA": (expand(np.load(meanabs_pool / "meanabs_ga_mlp_full.npy"), fi), ex("ga_scores.npy")),
    }


def corr(scores, effects: dict, importance: bool):
    s, v = [], []
    for f, d in effects.items():
        x = scores[int(f)]
        s.append(abs(x) if importance else x)
        v.append(abs(d) if importance else d)
    r, p = spearmanr(s, v)
    return float(r), float(p)


def stats_for(methods, unsigned, dabs, dsig, mode):
    oriented = {k: (-v if mode == "ablation" else v) for k, v in dsig.items()}
    out = {}
    for name, (imp, dire) in methods.items():
        r, p = corr(imp, dabs, True)
        out[f"{name}:importance"] = {"rho": r, "p": p}
        if name in unsigned:
            continue
        r, p = corr(dire, oriented, False)
        out[f"{name}:directional"] = {"rho": r, "p": p}
    return out


def render_run(header, methods, unsigned, stats, mode, bh):
    L = [header.rstrip("\n"), NOTE.rstrip("\n"),
         "Importance faithfulness  (Spearman rho of |attribution| vs |delta|):"]
    for name in methods:
        s = stats[f"{name}:importance"]
        L.append(f"  {name:14s}  rho={s['rho']:+.3f}  p={s['p']:.4f}")
    L.append(
        f"Directional faithfulness (Spearman rho of attribution vs "
        f"{'-delta' if mode == 'ablation' else 'delta'}; positive = faithful under mode={mode}):"
    )
    for name in methods:
        if name in unsigned:
            L.append(f"  {name:14s}  n/a (magnitude-only score, no direction)")
            continue
        s = stats[f"{name}:directional"]
        L.append(f"  {name:14s}  rho={s['rho']:+.3f}  p={s['p']:.4f}")
    if bh is not None:
        L += ["", f"Benjamini-Hochberg FDR correction (family = {len(bh)} tests in this run, alpha=0.05):"]
        for k, row in bh.items():
            L.append(f"  {k:28s}  p={row['p']:.4f}  q={row['q']:.4f} {'*' if row['reject'] else ' '}")
        L.append(f"  {sum(r['reject'] for r in bh.values())} of {len(bh)} survive FDR correction.")
    return "\n".join(L) + "\n"


def rescore_runs(arm, methods):
    for rel in RUN_DIRS[arm]:
        src = output_path(rel)
        tag = next(src.glob("model_deltas_*.json")).stem.removeprefix("model_deltas_")
        d = json.load(open(src / f"model_deltas_{tag}.json"))
        old = (src / f"faithfulness_{tag}.txt").read_text()
        header = old.split("Importance faithfulness")[0]
        had_bh = (src / f"faithfulness_{tag}_bh.json").exists()
        stats = stats_for(methods, UNSIGNED[arm], d["delta_abs"], d["delta_signed"], d["mode"])
        bh = benjamini_hochberg({k: v["p"] for k, v in stats.items()}) if had_bh else None
        dst = OUT_NEW / rel.split("/")[0] / rel.split("/")[1]
        dst.mkdir(parents=True, exist_ok=True)
        (dst / f"faithfulness_{tag}.txt").write_text(
            render_run(header, methods, UNSIGNED[arm], stats, d["mode"], bh)
        )
        if had_bh:
            corrected = bh
            json.dump(
                {
                    "mode": d["mode"], "selection": d["selection"], "k": len(d["delta_abs"]),
                    "family_size": len(corrected), "alpha": ALPHA,
                    "tests": {n: {**stats[n], **corrected[n]} for n in stats},
                },
                open(dst / f"faithfulness_{tag}_bh.json", "w"), indent=2,
            )
        print(f"[{arm}] {rel}: SHAP imp rho={stats['SHAP:importance']['rho']:+.3f} "
              f"dir rho={stats['SHAP:directional']['rho']:+.3f}")


def rescore_seed_sweep(arm, methods):
    sweep_dir, delta_dir = SEED_SWEEPS[arm]
    src = output_path(sweep_dir)
    jf = next(src.glob("seed_sweep_*.json"))
    pub = json.load(open(jf))
    tag = jf.stem.removeprefix("seed_sweep_")
    run = output_path(delta_dir)
    d = json.load(open(next(run.glob("model_deltas_*.json"))))
    seeds, mode = pub["seeds"], pub["mode"]

    # Validate: the all-114 run's deltas must reproduce the published sweep for the
    # *signed* scores it used (same alpha, rows and scope), or reuse is not exact.
    signed = {k: (v[1], v[1]) for k, v in methods.items()}
    signed["SHAP"] = (methods["SHAP"][1], methods["SHAP"][1])
    max_diff = 0.0
    per_seed, per_seed_check = {}, {}
    for s in seeds:
        feats = [str(int(f)) for f in np.load(src / f"candidates_seed{s}_{tag}.npy")]
        da = {f: d["delta_abs"][f] for f in feats}
        ds = {f: d["delta_signed"][f] for f in feats}
        chk = stats_for(signed, UNSIGNED[arm], da, ds, mode)
        for k, v in chk.items():
            max_diff = max(max_diff, abs(v["rho"] - pub["per_seed"][str(s)][k]["rho"]))
        per_seed[str(s)] = stats_for(methods, UNSIGNED[arm], da, ds, mode)
    print(f"[{arm}] seed-sweep reuse check: max |rho - published| = {max_diff:.2e}")
    assert max_diff < 1e-6, "saved all-114 deltas do not reproduce the published seed sweep"

    agg = _seed9a.aggregate(per_seed, seeds, pub["ref_seed"])
    # identical (rho, p) within a seed are one test (linear: SHAP = IG = GA)
    pooled_in = {}
    for s in seeds:
        seen = set()
        for k, v in per_seed[str(s)].items():
            sig = (round(v["rho"], 12), round(v["p"], 15))
            if sig not in seen:
                seen.add(sig)
                pooled_in[f"seed{s}:{k}"] = v["p"]
    pooled = benjamini_hochberg(pooled_in)
    n_rej = sum(r["reject"] for r in pooled.values())
    old_txt = (src / f"seed_sweep_{tag}.txt").read_text()
    keep = old_txt.split("Pooled Benjamini")[0]
    header = (
        keep + NOTE
        + f"Pooled Benjamini-Hochberg over all {len(pooled)} per-seed tests "
        f"(distinct tests, {len(seeds)} seeds): {n_rej} survive at FDR 0.05 "
        "(the tests overlap - shared features - so this count is descriptive)."
    )
    dst = OUT_NEW / sweep_dir
    dst.mkdir(parents=True, exist_ok=True)
    pub.update(per_seed=per_seed, aggregate=agg,
               pooled_bh={"family_size": len(pooled), "n_reject": int(n_rej), "tests": pooled})
    json.dump(pub, open(dst / f"seed_sweep_{tag}.json", "w"), indent=2)
    (dst / f"seed_sweep_{tag}.txt").write_text(_seed9a.render(agg, per_seed, seeds, header.rstrip("\n")))
    a = agg["SHAP:importance"]
    print(f"[{arm}] seed sweep SHAP imp: mean {a['mean']:+.3f} sd {a['std']:.3f} "
          f"min {a['min']:+.3f} max {a['max']:+.3f} frac>0 {a['frac_positive']:.2f}; pooled BH {n_rej}/{len(pooled)}")


def ranking_agreement():
    lines = []
    # Linear: every non-zero-weight feature.
    probe = joblib.load(
        Path(os.environ.get("SHAP_SAE_CHECKPOINTS", "checkpoints")) / "probe_layer_7.joblib"
    )
    w = np.asarray(probe.coef_[0], dtype=np.float64)
    support = np.load(output_path("8_rankings_recompute", "support.npy"))
    # mean |IG| = mean |GA| = mean |phi| per example for a linear probe (identity);
    # the full-width array covers every non-zero-weight feature, not just the 114.
    ig = ga = np.load(output_path("7c_global_meanabs", "meanabs_linear_full.npy"))[support]
    ma = np.load(output_path("7c_global_meanabs", "meanabs_linear_full.npy"))[support]
    lines.append(f"Linear arm ranking agreement, |score|, n={len(support)} non-zero-weight features; SHAP = mean |phi| (full train bg)")
    for a, b, na, nb in ((w[support], ma, "Probe", "SHAP"), (ig, ma, "IG", "SHAP"), (ma, ga, "SHAP", "GA"),
                         (w[support], ig, "Probe", "IG"), (ig, ga, "IG", "GA"), (w[support], ga, "Probe", "GA")):
        r, p = spearmanr(np.abs(a), np.abs(b))
        lines.append(f"  {na} vs {nb}: rho={r:.3f} p={p:.4g}")
    # MLP
    rk = output_path("13_mlp_ranking")
    sc = {"probe": np.load(rk / "probe_scores.npy")}
    for n in ("ig", "ga"):
        sc[n] = np.load(output_path("7c_global_meanabs", f"meanabs_{n}_mlp_full.npy"))
    sc["shap"] = np.load(output_path("7c_global_meanabs", "meanabs_mlp.npy"))
    lines.append(f"\nMLP arm ranking agreement, |score|, n={len(sc['shap'])}; SHAP = mean |phi| (full train bg)")
    names = {"probe": "Probe", "ig": "IG", "ga": "GA", "shap": "SHAP"}
    ks = list(names)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            r, p = spearmanr(np.abs(sc[ks[i]]), np.abs(sc[ks[j]]))
            lines.append(f"  {names[ks[i]]} vs {names[ks[j]]}: rho={r:.3f} p={p:.4g}")
    return "\n".join(lines) + "\n"


def main():
    arms = sys.argv[1:] or ["linear", "mlp"]
    for arm in arms:
        methods = load_methods(arm)
        rescore_runs(arm, methods)
        rescore_seed_sweep(arm, methods)
    if "mlp" not in arms:
        return
    # New-root MLP ranking dir: SHAP score := mean |phi| (signed kept alongside)
    rk_new = OUT_NEW / "13_mlp_ranking"
    rk_old = output_path("13_mlp_ranking")
    rk_new.mkdir(parents=True, exist_ok=True)
    np.save(rk_new / "shap_scores_signed.npy", np.load(rk_old / "shap_scores.npy"))
    np.save(rk_new / "shap_scores.npy", np.load(output_path("7c_global_meanabs", "meanabs_mlp.npy")))
    for n in ("ig", "ga"):  # mean |.| (full-train baseline) replaces the signed means; signed kept
        np.save(rk_new / f"{n}_scores_signed.npy", np.load(rk_old / f"{n}_scores.npy"))
        np.save(rk_new / f"{n}_scores.npy", np.load(output_path("7c_global_meanabs", f"meanabs_{n}_mlp_full.npy")))
    text = ranking_agreement()
    (OUT_NEW / "ranking_agreement_meanabs.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
