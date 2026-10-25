"""
Diagnostic for 9.5_tuning: is the layer-11 residual probe already saturated
before any steering? Reuses 9.5_tuning's own functions on its exact pilot rows
and pilot features; changes nothing in src/ or in 9.5_tuning's outputs.

Writes run3/outputs/9.5_tuning_diag/.
"""

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from sae_lens import SAE

REPO = Path("/home/ubuntu/shap-sae")
sys.path.insert(0, str(REPO / "src"))
from utils import checkpoint_path, load_model, load_splits, output_path  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tune95", REPO / "src/global_linear/9.5_tuning.py"
)
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)
steer9 = tune.steer9

OUT = output_path("9.5_tuning_diag")
OUT.mkdir(parents=True, exist_ok=True)
TUNE = output_path("9.5_tuning")

pilot_rows = np.load(TUNE / "pilot_row_indices.npy")
median_scale = float(np.median(np.load(TUNE / "candidate_scales.npy")))
pilot_features = tune.load_ranking_candidates()["pilot_features"]

model = load_model()
sae = SAE.from_pretrained(
    "gpt2-small-resid-post-v5-32k", f"blocks.{tune.LAYER}.hook_resid_post"
).to(model.cfg.device)
readout_layer = model.cfg.n_layers - 1
residual_probe = joblib.load(
    checkpoint_path(f"residual_probe_layer_{readout_layer}.joblib")
)

train_ds, _, _ = load_splits()
sents = list(train_ds["sentence"])
pilot_tokens = model.to_tokens([sents[int(i)] for i in pilot_rows])
pilot_labels = np.asarray(train_ds["label"])[pilot_rows]


def sat(p):
    return (p < 0.02) | (p > 0.98)


# 1. SAE-recon baseline (the one 9.5_tuning compares against)
base = tune.compute_recon_baseline_probs(
    model, sae, residual_probe, pilot_tokens, readout_layer
)

# 2. Clean forward (no SAE), for context
store = {}


def cap(resid, hook):
    store["r"] = resid
    return resid


clean = []
for t in pilot_tokens:
    t = steer9._as_batch(t, model.cfg.device)
    with torch.no_grad(), model.hooks(
        fwd_hooks=[(f"blocks.{readout_layer}.hook_resid_post", cap)]
    ):
        model(t)
    pooled = steer9.pool_last_non_pad_token(model, t, store["r"])
    clean.append(float(residual_probe.predict_proba(pooled.reshape(1, -1))[0, 1]))
clean = np.asarray(clean)

# 3. Per-alpha: total saturation (9.5_tuning's metric) and newly-saturated
#    (saturated after steering but not at baseline), on a grid extended downward.
mults = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 0.5, 1.0, 2.0, 4.0)
rows = []
for m in mults:
    alpha = m * median_scale
    r = tune.pilot_effects_for_alpha(
        model, sae, residual_probe, pilot_tokens, pilot_features,
        alpha=alpha, readout_layer=readout_layer, baseline_probs=base,
    )
    # pilot_effects_for_alpha does not return per-example probs; recompute the
    # newly-saturated fraction with the same hook it uses.
    new_sat = []
    for feat in pilot_features:
        def hook(resid, hook, f=int(feat), a=alpha):
            acts = sae.encode(resid).clone()
            acts[..., f] = acts[..., f] + a
            return sae.decode(acts)

        ps = []
        for t in pilot_tokens:
            t = steer9._as_batch(t, model.cfg.device)
            with torch.no_grad(), model.hooks(fwd_hooks=[
                (f"blocks.{tune.LAYER}.hook_resid_post", hook),
                (f"blocks.{readout_layer}.hook_resid_post", cap),
            ]):
                model(t)
            pooled = steer9.pool_last_non_pad_token(model, t, store["r"])
            ps.append(float(residual_probe.predict_proba(pooled.reshape(1, -1))[0, 1]))
        ps = np.asarray(ps)
        new_sat.append(sat(ps) & ~sat(base))
    rows.append({
        "multiplier": m,
        "alpha": alpha,
        "mean_abs_dp": r["mean_abs_dp"],
        "mean_signed_dp": r["mean_signed_dp"],
        "sat_frac_total": r["saturation_frac"],
        "sat_frac_new": float(np.mean(np.concatenate(new_sat))),
    })
    print(rows[-1], flush=True)

summary = {
    "n_pilot": int(len(pilot_rows)),
    "pilot_features": [int(f) for f in pilot_features],
    "median_scale": median_scale,
    "baseline_recon": {
        "sat_frac": float(sat(base).mean()),
        "frac_p_lt_0.02": float((base < 0.02).mean()),
        "frac_p_gt_0.98": float((base > 0.98).mean()),
        "probe_acc_on_pilot": float(((base > 0.5) == pilot_labels).mean()),
        "quantiles": dict(zip(["p5", "p25", "p50", "p75", "p95"],
                              np.quantile(base, [.05, .25, .5, .75, .95]).tolist())),
    },
    "baseline_clean": {
        "sat_frac": float(sat(clean).mean()),
        "probe_acc_on_pilot": float(((clean > 0.5) == pilot_labels).mean()),
    },
    "grid": rows,
}
(OUT / "saturation_diag.json").write_text(json.dumps(summary, indent=2))

lines = [
    "9.5_tuning saturation diagnostic (probe readout, same pilot rows/features)",
    f"n_pilot={summary['n_pilot']}  pilot_features={summary['pilot_features']}  "
    f"median_scale={median_scale:.4f}",
    "",
    f"baseline (SAE recon) sat_frac = {summary['baseline_recon']['sat_frac']:.3f}  "
    f"(P<0.02: {summary['baseline_recon']['frac_p_lt_0.02']:.3f}, "
    f"P>0.98: {summary['baseline_recon']['frac_p_gt_0.98']:.3f})",
    f"baseline (clean)     sat_frac = {summary['baseline_clean']['sat_frac']:.3f}",
    f"probe acc on pilot rows: recon {summary['baseline_recon']['probe_acc_on_pilot']:.3f}, "
    f"clean {summary['baseline_clean']['probe_acc_on_pilot']:.3f}",
    "",
    f"{'mult':>7} {'alpha':>9} {'mean|dP|':>9} {'mean dP':>9} {'sat_tot':>8} {'sat_new':>8}",
]
for r in rows:
    lines.append(
        f"{r['multiplier']:7.4f} {r['alpha']:9.4f} {r['mean_abs_dp']:9.4f} "
        f"{r['mean_signed_dp']:+9.4f} {r['sat_frac_total']:8.3f} {r['sat_frac_new']:8.3f}"
    )
(OUT / "saturation_diag.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
