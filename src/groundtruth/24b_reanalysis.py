"""
Step 27b — threshold-sensitivity and readout-reliability reanalysis of the
minimal-pair run (answers the "moved" threshold / shaky-readout critique).

Reuses 24_minimal_pairs.py's pair builders and seed order, so it rebuilds the
*same* pairs as the committed run (checked against per_pair_<tag>.json). It only
does two clean forward passes per pair plus one cheap null edit — no per-feature
ablations — so it is a couple of minutes.

Reports, per edit type:
  A. Sweep of the "moved" threshold (relative and absolute): AUROC for each
     method AND for activity-only baselines (|act| at the edited last token,
     active/inactive) and the oracle |delta act|. If local SHAP only matches
     the activity baselines, its AUROC reflects "which features fire", not
     attribution quality.
  B. Threshold-free: Spearman of |score| vs |delta act| per pair.
  C. AUROC restricted to candidates active at the edited last token.
  D. Readout: Wilson CI on sign accuracy, sign accuracy by |text_delta| bin,
     and the |text_delta| of a null edit (neutral word) as a noise floor.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE
from scipy.stats import spearmanr

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    CANONICAL_SEED,
    checkpoint_path,
    load_eval_and_background,
    load_model,
    load_splits,
    output_path,
)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mp = _load("minpairs24", Path(__file__).resolve().parent / "24_minimal_pairs.py")

REL_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)
ABS_THRESHOLDS = (0.1, 0.5, 1.0)
METHODS = ("local_SHAP", "global_SHAP", "Probe weights")  # IG/GA == global_SHAP here
BASELINES = ("act_edited", "active_edited", "oracle_dact")
N_BOOT = 2000


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def boot_ci(x: np.ndarray, rng) -> list[float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return [float("nan")] * 3
    m = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(axis=1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def null_pairs(sentences, edit, rng, n):
    """Neutral edits with no intended sentiment change, for a readout noise floor."""
    flags = mp._simplicity.lexicon_flags(sentences)
    neutral = [s for s, ok in zip(sentences, ~flags["has_any"]) if ok and len(s.split()) >= 4]
    out = []
    for i in rng.permutation(len(neutral))[:n]:
        s = neutral[int(i)]
        if edit == "insert":
            e = s.rstrip() + " it was here ." + s[len(s.rstrip()) :]
        elif edit == "insert-interior":
            parts = s.split()
            m = len(parts) // 2
            e = " ".join(parts[:m] + ["it", "was", "here,"] + parts[m:]) + s[len(s.rstrip()) :]
        else:
            parts = s.split()
            j = int(rng.integers(1, len(parts) - 1))  # interior word
            e = " ".join(parts[:j] + parts[j + 1 :]) + s[len(s.rstrip()) :]
        out.append((s, e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--edit", choices=tuple(mp.BUILDERS), required=True)
    ap.add_argument("--n-pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    ap.add_argument("--k-moved", type=int, default=15)
    ap.add_argument("--k-distractor", type=int, default=15)
    ap.add_argument("--out-dir", type=Path, default=output_path("24_minimal_pairs"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    _, _, shap_ds = load_splits()
    sentences = list(shap_ds["sentence"])
    builder = mp.BUILDERS[args.edit]
    pairs = builder(sentences, rng, args.n_pairs)
    feature_indices = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    distractors = rng.choice(
        feature_indices, size=min(args.k_distractor, len(feature_indices)), replace=False
    )

    tag = f"{args.edit}_n{args.n_pairs}"
    saved = json.load(open(args.out_dir / f"per_pair_{tag}.json"))
    assert len(saved) == len(pairs), "pair count differs from committed run"

    probe = joblib.load(checkpoint_path("probe_layer_7.joblib"))
    probe.verbose = 0
    phi = np.load(output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy")).astype(np.float64)
    probe_w = np.asarray(probe.coef_[0], dtype=np.float64)
    _, background, _, _ = load_eval_and_background(layer=mp.LAYER)
    background_mean = background.mean(axis=0, keepdims=True)

    model = load_model()
    sae = SAE.from_pretrained(mp.SAE_RELEASE, mp.SAE_ID).to(model.cfg.device)
    pos_id, neg_id = mp._steer9.sentiment_token_ids(model)
    point = f"blocks.{mp.LAYER}.hook_resid_post"

    def recon(resid, hook):
        return sae.decode(sae.encode(resid))

    def logitdiff(tok):
        return mp.score_with_hooks(model, tok, pos_id, neg_id, [(point, recon)])

    # --- per-pair raw quantities -------------------------------------------------
    recs = []
    for k, pair in enumerate(pairs):
        tok_o = model.to_tokens([pair["original"]])
        tok_e = model.to_tokens([pair["edited"]])
        base_o, base_e = logitdiff(tok_o), logitdiff(tok_e)
        # Fail loudly if this is not the committed run's pair.
        assert saved[k]["word"] == pair["word"], f"pair {k}: word mismatch"
        assert abs(saved[k]["logitdiff_edited"] - base_e) < 1e-3, f"pair {k}: logit mismatch"
        a_o = mp.last_token_sae_acts(model, sae, tok_o)
        a_e = mp.last_token_sae_acts(model, sae, tok_e)
        d = a_e - a_o
        moved = np.argsort(np.abs(d))[::-1][: args.k_moved]
        cand = np.unique(np.concatenate([moved, distractors]))
        local = np.zeros(len(phi))
        local[feature_indices] = np.asarray(
            mp._shap7.run_linearshap(probe, a_e[None, :], background_mean, feature_indices)
        )[0]
        recs.append(
            dict(
                cand=cand,
                d=np.abs(d[cand]),
                act_e=np.abs(a_e[cand]),
                active_e=(a_e[cand] > 0).astype(float),
                scores={
                    "local_SHAP": np.abs(local[cand]),
                    "global_SHAP": np.abs(phi[cand]),
                    "Probe weights": np.abs(probe_w[cand]),
                    "act_edited": np.abs(a_e[cand]),
                    "active_edited": (a_e[cand] > 0).astype(float),
                    "oracle_dact": np.abs(d[cand]),
                },
                text_delta=base_e - base_o,
                expected_sign=pair["expected_sign"],
            )
        )

    brng = np.random.default_rng(0)
    names = METHODS + BASELINES

    def sweep(label_fn, subset=None):
        res = {}
        for n in names:
            vals = []
            for r in recs:
                keep = np.ones(len(r["cand"]), bool) if subset is None else subset(r)
                if keep.sum() < 3:
                    continue
                lab = label_fn(r)[keep]
                vals.append(mp.auroc(lab, r["scores"][n][keep]))
            res[n] = boot_ci(np.array(vals), brng)
        return res

    out = {"edit": args.edit, "n_pairs": len(recs), "seed": args.seed}
    out["rel_threshold"] = {
        str(t): sweep(lambda r, t=t: r["d"] >= t * r["d"].max()) for t in REL_THRESHOLDS
    }
    out["abs_threshold"] = {
        str(t): sweep(lambda r, t=t: r["d"] >= t) for t in ABS_THRESHOLDS
    }
    out["active_only_rel0.10"] = sweep(
        lambda r: r["d"] >= 0.10 * r["d"].max(), subset=lambda r: r["active_e"] > 0
    )
    out["frac_moved_rel0.10"] = float(
        np.mean([(r["d"] >= 0.10 * r["d"].max()).mean() for r in recs])
    )
    out["frac_candidates_active_edited"] = float(np.mean([r["active_e"].mean() for r in recs]))

    out["spearman_vs_dact"] = {}
    for n in METHODS + ("act_edited", "active_edited"):
        v = [spearmanr(r["scores"][n], r["d"]).statistic for r in recs]
        out["spearman_vs_dact"][n] = boot_ci(np.array(v, float), brng)

    # --- readout reliability -----------------------------------------------------
    td = np.array([r["text_delta"] for r in recs])
    ok = np.sign(td) == np.array([r["expected_sign"] for r in recs])
    lo, hi = wilson(int(ok.sum()), len(ok))
    nulls = null_pairs(sentences, args.edit, np.random.default_rng(args.seed + 1), 100)
    null_abs = np.array(
        [abs(logitdiff(model.to_tokens([e])) - logitdiff(model.to_tokens([o]))) for o, e in nulls]
    )
    floor = float(np.percentile(null_abs, 90))
    big = np.abs(td) >= floor
    out["readout"] = {
        "sign_acc": float(ok.mean()),
        "sign_acc_wilson95": [lo, hi],
        "null_abs_delta_quantiles": {
            q: float(np.percentile(null_abs, q)) for q in (50, 75, 90, 95)
        },
        "noise_floor_p90": floor,
        "n_above_floor": int(big.sum()),
        "sign_acc_above_floor": float(ok[big].mean()) if big.any() else float("nan"),
        "sign_acc_above_floor_wilson95": wilson(int(ok[big].sum()), int(big.sum()))
        if big.any()
        else None,
        "sign_acc_by_abs_delta_tertile": [
            float(ok[(np.abs(td) >= lo_) & (np.abs(td) <= hi_)].mean())
            for lo_, hi_ in zip(
                np.percentile(np.abs(td), [0, 33.3, 66.7]),
                np.percentile(np.abs(td), [33.3, 66.7, 100]),
            )
        ],
    }
    # AUROC / rho on the above-floor subset, from committed per-pair numbers.
    out["readout"]["above_floor_subset_from_committed"] = {
        m: {
            "auroc": float(np.nanmean([saved[i]["methods"][m]["auroc_moved"] for i in np.where(big)[0]])),
            "rho": float(np.nanmean([saved[i]["methods"][m]["rho_recovery"] for i in np.where(big)[0]])),
        }
        for m in ("local_SHAP", "global_SHAP", "Probe weights")
    } if big.any() else {}

    (args.out_dir / f"reanalysis_{tag}.json").write_text(json.dumps(out, indent=2))

    def f(c):
        return f"{c[0]:.3f} [{c[1]:.3f},{c[2]:.3f}]"

    L = [f"Minimal-pair reanalysis  edit={args.edit}  n={len(recs)}", "=" * 70]
    L.append(f"frac candidates 'moved' @rel0.10 = {out['frac_moved_rel0.10']:.2f};  "
             f"frac candidates active in edited = {out['frac_candidates_active_edited']:.2f}")
    for title, blocks in (("rel", out["rel_threshold"]), ("abs", out["abs_threshold"])):
        for t, res in blocks.items():
            L.append(f"\nAUROC, moved = |dact| >= {t}{'*max' if title == 'rel' else ''}")
            for n in names:
                L.append(f"  {n:15s} {f(res[n])}")
    L.append("\nAUROC, rel0.10, candidates active at edited last token only")
    for n in names:
        L.append(f"  {n:15s} {f(out['active_only_rel0.10'][n])}")
    L.append("\nSpearman(|score|, |dact|) per pair (threshold-free)")
    for n, c in out["spearman_vs_dact"].items():
        L.append(f"  {n:15s} {f(c)}")
    r = out["readout"]
    L.append("\nReadout")
    L.append(f"  sign acc {r['sign_acc']:.3f}  Wilson95 [{lo:.3f},{hi:.3f}]")
    L.append(f"  null-edit |delta| quantiles {r['null_abs_delta_quantiles']}")
    L.append(f"  pairs with |delta| >= null p90 ({floor:.3f}): {r['n_above_floor']}, "
             f"sign acc {r['sign_acc_above_floor']:.3f}")
    L.append(f"  sign acc by |delta| tertile: {r['sign_acc_by_abs_delta_tertile']}")
    L.append(f"  above-floor subset (committed per-pair): {r['above_floor_subset_from_committed']}")
    (args.out_dir / f"reanalysis_{tag}.txt").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
