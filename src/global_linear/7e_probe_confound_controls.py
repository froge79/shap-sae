"""
Probe-vs-LM confound controls for the global linear arm (CPU, closed form).

The faithfulness test correlates an attribution of the PROBE with an intervention
on the LANGUAGE MODEL, so a low rho cannot say whether the attribution misdescribes
the probe or the probe misdescribes the LM. For a linear probe the probe's own
response to the intervention is closed form, which lets the two be separated:

  probe steering effect   P_j = w_j * alpha_j          (a_j <- a_j + alpha_j)
  probe ablation effect   A_j = -mean_e(w_j * x_ej)     (a_j <- 0)
  LM effect               D_j = measured delta_signed   (9_steer, n_eval rows)

Scores compared (all on the 114-feature filtered axis, signed unless noted):
  phi_bg100        published Phi, canonical 100-row train background
  phi_bgfull       Phi against every train row
  phi_classmatched Phi against a background whose class mix equals the eval rows'
                   (p_pos * mu_pos + p_neg * mu_neg over the full train set, so no
                   sampling noise). Identical to the eval-proportion-weighted
                   WITHIN-class shift, sum_c p_c w_j (mean_e,c x_j - mu_c,j).
  wx               mean_e(w_j x_ej), NO background (= SHAP with a zero baseline,
                   = -A_j)
  weight           w_j
  meanabs          mean|w_j (x_ej - mu_j)|, full train background (importance only)

Tables:
  1. score vs LM effect D        (the published test, same rho as 9_steer)
  2. score vs probe effect P     (attribution error only: no LM involved)
  3. P vs D                      (probe-LM mismatch: the ceiling for ANY probe-based score)
  4. sign agreement with w, and the class-matched bootstrap (500 stratified
     100-row backgrounds, mirroring the unstratified bootstrap in the paper)
Ablation (k=20 random run, if present) is scored against A_j the same way.

Output: 7e_probe_confound_controls/{summary.txt, summary.json}
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
N_BOOT = 500
N_BG = 100


def rho(a, b, *, importance: bool = False) -> float:
    a, b = np.asarray(a), np.asarray(b)
    if importance:
        a, b = np.abs(a), np.abs(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b)[0])


def boot_ci(a, b, *, importance: bool, rng, n=2000) -> tuple[float, float]:
    """Percentile CI for rho, resampling features. Descriptive: n features is small."""
    a, b = np.asarray(a), np.asarray(b)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        r = rho(a[i], b[i], importance=importance)
        if not np.isnan(r):
            out.append(r)
    return tuple(np.percentile(out, [2.5, 97.5]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--steer-dir",
        default=str(output_path("9_steer", "steering_all_scaled")),
        help="dir holding model_deltas_steering_all_k114.json",
    )
    ap.add_argument(
        "--ablation-json",
        default=str(
            output_path("9_steer", "ablation_random_k20_scaled", "model_deltas_ablation_random_k20.json")
        ),
    )
    ap.add_argument("--out-dir", default=str(output_path("7e_probe_confound_controls")))
    args = ap.parse_args()

    probe = joblib.load(checkpoint_path(f"probe_layer_{LAYER}.joblib"))
    fi = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    w = np.asarray(probe.coef_[0], dtype=np.float64)[fi]

    x_full, bg100, eval_idx, bg100_idx = load_eval_and_background(layer=LAYER, n_eval=DEFAULT_N_EVAL)
    x = x_full[:, fi].astype(np.float64)
    shap_y = load_labels("shap", LAYER)[eval_idx].astype(int)
    train = load_activations("train", LAYER, mmap=True)
    train_y = load_labels("train", LAYER).astype(int)
    xt = np.asarray(train[:, fi], dtype=np.float64)
    mu_full = xt.mean(0)
    mu_pos, mu_neg = xt[train_y == 1].mean(0), xt[train_y == 0].mean(0)
    p_pos = float(shap_y.mean())

    # ---- sanity: closed form reproduces the saved published Phi -----------------
    saved = np.load(output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy"))[fi]
    xbar = x.mean(0)
    phi100 = w * (xbar - bg100[:, fi].astype(np.float64).mean(0))
    err = float(np.abs(phi100 - saved).max())
    assert err < 1e-5, f"closed-form Phi disagrees with saved Phi (max err {err:.2e})"

    # ---- scores -----------------------------------------------------------------
    mu_cm = p_pos * mu_pos + (1 - p_pos) * mu_neg
    scores = {
        "phi_bg100": phi100,
        "phi_bgfull": w * (xbar - mu_full),
        "phi_classmatched": w * (xbar - mu_cm),
        "wx": w * xbar,
        "weight": w.copy(),
    }
    meanabs = np.abs(w * (x - mu_full)).mean(0)

    # ---- probe-side effects -----------------------------------------------------
    steer_json = json.load(open(Path(args.steer_dir) / "model_deltas_steering_all_k114.json"))
    keys = [str(int(j)) for j in fi]
    D = np.array([steer_json["delta_signed"][k] for k in keys])
    Dabs = np.array([steer_json["delta_abs"][k] for k in keys])  # mean|diff|, the importance target
    alpha = np.array([steer_json["alpha_by_feature"][k] for k in keys])
    P = w * alpha
    A = -(w * xbar)  # probe ablation effect (signed, mean over eval rows)

    lines: list[str] = []
    res: dict = {"n_features": len(fi), "p_pos_eval": p_pos, "closed_form_max_err": err}
    rng = np.random.default_rng(CANONICAL_SEED)

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("Probe-vs-LM confound controls (linear arm, signed steering, n=%d features)" % len(fi))
    emit(f"eval positive fraction {p_pos:.3f}; train positive fraction {train_y.mean():.3f}")
    emit(f"steer run: {steer_json['mode']} scope={steer_json['intervene_scope']} "
         f"n_eval={steer_json['n_eval']}")
    emit("CAVEAT: intervention is all-token, the probe reads the last token only, so the")
    emit("closed-form P_j = w_j*alpha_j is the DIRECT last-token effect; it omits")
    emit("propagation through the LM, which is exactly the mismatch Table 3 measures.")
    emit("Features are the same pool for every row; CIs resample features (descriptive).")
    emit("")

    def table(title, targets, key):
        emit(title)
        emit(f"  {'score':<18}" + "".join(f"{t:>26}" for t in targets))
        res[key] = {}
        for name, s in list(scores.items()) + [("meanabs(imp only)", meanabs)]:
            row, cells = {}, ""
            for tname, (tv, tabs) in targets.items():
                if name == "meanabs(imp only)":
                    r_dir = float("nan")
                else:
                    r_dir = rho(s, tv)
                r_imp = rho(s, tabs, importance=True)
                row[tname] = {"directional": r_dir, "importance": r_imp}
                cells += f"{f'dir {r_dir:+.3f} / imp {r_imp:+.3f}':>26}"
            res[key][name] = row
            emit(f"  {name:<18}{cells}")
        emit("")

    table(
        "Table 1. score vs LM effect D (the published test)",
        {"D(steer)": (D, Dabs)},
        "score_vs_LM",
    )
    table(
        "Table 2. score vs PROBE effect P=w*alpha (attribution error only, no LM)",
        {"P(steer)": (P, P)},
        "score_vs_probe",
    )

    emit("Table 3. probe-LM mismatch: closed-form probe effect vs measured LM effect")
    t3 = {}
    for nm, (a, b) in {
        "P vs D (signed)": (P, D),
        "|P| vs mean|D|": (np.abs(P), Dabs),
        "-A vs D (signed)": (-A, D),
    }.items():
        r = rho(a, b)
        lo, hi = boot_ci(a, b, importance=False, rng=rng)
        t3[nm] = {"rho": r, "ci95": [lo, hi]}
        emit(f"  {nm:<18} rho={r:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    res["probe_vs_LM"] = t3
    emit("  Read: if P-vs-D is itself low, a score that EXACTLY reproduces the probe cannot")
    emit("  score higher against D, so the LM test cannot separate attribution error from")
    emit("  probe mismatch. Compare Table 1 rows against this ceiling, not against 1.0.")
    emit("  (weight == sign-only version of P up to alpha scaling; P adds the alpha_j scale.)")
    emit("")

    # ---- sign agreement with the probe weight ----------------------------------
    emit("Table 4. sign agreement of Phi with sign(w) over the 114 pool, and over top-20 |Phi| within the 114-feature pool")
    res["sign_agreement"] = {}
    for name in ("phi_bg100", "phi_bgfull", "phi_classmatched", "wx"):
        s = scores[name]
        agree = float((np.sign(s) == np.sign(w)).mean())
        top = np.argsort(-np.abs(s))[:20]
        agree20 = float((np.sign(s[top]) == np.sign(w[top])).mean())
        res["sign_agreement"][name] = {"all": agree, "top20_abs": agree20}
        emit(f"  {name:<18} all {agree:.3f}   pool top-20 |score| {agree20:.3f}")
    emit("")

    # class-matched bootstrap: stratified 100-row train backgrounds
    pos_idx, neg_idx = np.where(train_y == 1)[0], np.where(train_y == 0)[0]
    n_pos = int(round(p_pos * N_BG))
    ag, rd = [], []
    for _ in range(N_BOOT):
        bi = np.concatenate([rng.choice(pos_idx, n_pos, replace=False),
                             rng.choice(neg_idx, N_BG - n_pos, replace=False)])
        ph = w * (xbar - xt[bi].mean(0))
        ag.append(float((np.sign(ph) == np.sign(w)).mean()))
        rd.append(rho(ph, D))
    ru = []  # unstratified, for a like-for-like comparison
    agu = []
    for _ in range(N_BOOT):
        bi = rng.choice(len(xt), N_BG, replace=False)
        ph = w * (xbar - xt[bi].mean(0))
        agu.append(float((np.sign(ph) == np.sign(w)).mean()))
        ru.append(rho(ph, D))
    q = lambda v: [float(np.mean(v))] + [float(t) for t in np.percentile(v, [2.5, 97.5])]
    res["bootstrap"] = {
        "stratified": {"sign_agree": q(ag), "directional_rho_vs_D": q(rd)},
        "unstratified": {"sign_agree": q(agu), "directional_rho_vs_D": q(ru)},
        "n_boot": N_BOOT, "n_bg": N_BG, "n_pos_in_bg": n_pos,
    }
    emit(f"Table 5. {N_BOOT} random 100-row backgrounds (mean [2.5%, 97.5%])")
    for lab, a, r in (("stratified (class-matched)", ag, rd), ("unstratified", agu, ru)):
        emit(f"  {lab:<28} sign-agree {q(a)[0]:.3f} [{q(a)[1]:.3f},{q(a)[2]:.3f}]   "
             f"dir rho vs D {q(r)[0]:+.3f} [{q(r)[1]:+.3f},{q(r)[2]:+.3f}]")
    emit("")

    # ---- ablation (k=20 random) -------------------------------------------------
    abl_path = Path(args.ablation_json)
    if abl_path.exists():
        abl = json.load(open(abl_path))
        ak = [k for k in abl["delta_signed"] if k in set(keys)]
        pos = {k: i for i, k in enumerate(keys)}
        ii = np.array([pos[k] for k in ak])
        Da = np.array([abl["delta_signed"][k] for k in ak])
        Daa = np.array([abl["delta_abs"][k] for k in ak])
        emit(f"Table 6. ABLATION, {abl['selection']} k={abl['k']} (n={len(ak)} features{', very small n' if len(ak) < 50 else ''}; scope={abl['intervene_scope']})")
        emit("  LM ablation effect vs closed-form probe ablation effect A_j = -mean(w*x):")
        res["ablation"] = {"n": len(ak)}
        ra = rho(A[ii], Da)
        ria = rho(A[ii], Daa, importance=True)
        lo, hi = boot_ci(A[ii], Da, importance=False, rng=rng)
        emit(f"  A vs D_ablation     dir {ra:+.3f} [{lo:+.3f},{hi:+.3f}]  imp {ria:+.3f}")
        res["ablation"]["A_vs_D"] = {"directional": ra, "importance": ria, "ci95_directional": [lo, hi]}
        for name, s in scores.items():
            # attribution predicts a NEGATIVE ablation effect, so orient D as in 9_steer
            r_d, r_i = rho(s[ii], -Da), rho(s[ii], Daa, importance=True)
            res["ablation"][name] = {"directional": r_d, "importance": r_i}
            emit(f"  {name:<18} vs D_abl  dir {r_d:+.3f}  imp {r_i:+.3f}")
        emit("")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    json.dump(res, open(out / "summary.json", "w"), indent=2)
    np.savez(out / "vectors.npz", feature_indices=fi, w=w, alpha=alpha, P=P, A=A, D=D, Dabs=Dabs,
             **{k: v for k, v in scores.items()})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
