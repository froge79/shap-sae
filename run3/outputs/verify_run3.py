"""
CLAUDE.md "Verification after the run", items 1-10, for run3 (n_eval=1000).
Run from the repo root with SHAP_SAE_OUTPUTS=run3/outputs SHAP_SAE_CHECKPOINTS=run3/checkpoints.
Writes run3/outputs/verification.txt. Exit code 1 if any hard check FAILs.
"""

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

REPO = Path("/home/ubuntu/shap-sae")
sys.path.insert(0, str(REPO / "src"))
from utils import sample_eval_rows  # noqa: E402

O = REPO / "run3/outputs"
C = REPO / "run3/checkpoints"
SCRATCH = Path(sys.argv[1])  # scratch root for item 8
N_MASK = len(np.load(O / "3_shap/shap_feature_indices.npy"))

out: list[str] = []
fails: list[str] = []


def say(s=""):
    out.append(s)
    print(s, flush=True)


def check(name, ok, detail=""):
    say(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        fails.append(name)


def rel(p):
    return str(Path(p).relative_to(REPO))


say("run3 verification (CLAUDE.md 'Verification after the run', items 1-10)")
say(f"outputs={rel(O)}  checkpoints={rel(C)}  n_eval=1000  mask size N={N_MASK}")
say()

# ---------------------------------------------------------------- 1
say("1. Split discipline")
ev, bg = sample_eval_rows(n_eval=1000)
evs = sorted(O.rglob("eval_indices*.npy"))
for p in evs:
    a = np.load(p)
    check(f"{rel(p)} is a prefix of eval_idx", a.ndim == 1 and np.array_equal(a, ev[: len(a)]),
          f"len={len(a)}")
for p in sorted(O.rglob("example_indices.npy")):
    a = np.load(p)
    check(f"{rel(p)} is a prefix of eval_idx (local scripts)",
          np.array_equal(a, ev[: len(a)]), f"len={len(a)}")
bgs = sorted(O.rglob("background_indices*.npy"))
for p in bgs:
    a = np.load(p)
    check(f"{rel(p)} == bg_idx", np.array_equal(a, bg), f"len={len(a)}")
say(f"  ({len(evs)} eval_indices files, {len(bgs)} background_indices files)")
say()

# ---------------------------------------------------------------- 2
say("2. Rows agree")
for a, b in [("7_shap_recompute/eval_indices_layer7.npy", "8_rankings_recompute/eval_indices.npy"),
             ("12_mlp_shap/eval_indices_layer7.npy", "13_mlp_ranking/eval_indices.npy")]:
    check(f"{a} == {b}", np.array_equal(np.load(O / a), np.load(O / b)))
say()

# ---------------------------------------------------------------- 3
say("3. Index spaces")
m3 = np.load(O / "3_shap/shap_feature_indices.npy")
m12 = np.load(O / "12_mlp_shap/shap_feature_indices.npy")
check(f"3_shap and 12_mlp_shap masks both ({N_MASK},) and equal",
      m3.shape == (N_MASK,) and m12.shape == (N_MASK,) and np.array_equal(m3, m12),
      f"{m3.shape} vs {m12.shape}")
m11 = np.load(O / "11_mlp_probe/feature_indices_layer_7.npy")
check("11_mlp_probe feature_indices == mask", np.array_equal(m11, m3))
check("9.5_tuning pool_indices == mask", np.array_equal(np.load(O / "9.5_tuning/pool_indices.npy"), m3))
ex = np.load(O / "7_shap_recompute/explained_feature_indices.npy")
check("7_shap_recompute/explained_feature_indices == arange(32768)", np.array_equal(ex, np.arange(32768)))
wide = [rel(p) for p in O.rglob("shap_feature_indices.npy") if np.load(p).shape == (32768,)]
check("no shap_feature_indices.npy of shape (32768,)", not wide, ", ".join(wide))
say()

# ---------------------------------------------------------------- 4
say("4. M1 identity (linear arm)")
corr8 = (O / "8_rankings_recompute/correlation.txt").read_text()
check("correlation.txt identity verdict PASS",
      re.search(r"^\s*PASS\b", corr8, re.M) is not None and re.search(r"^\s*FAIL\b", corr8, re.M) is None)
for pair in ("IG vs SHAP", "IG vs GA", "SHAP vs GA"):
    m = re.search(rf"{pair}:\s*ρ=([+-]?[\d.]+)", corr8)
    check(f"{pair} rho = 1.000", m is not None and m.group(1) == "1.000", m.group(0) if m else "missing")
say("  " + " | ".join(l.strip() for l in corr8.splitlines() if "max|" in l))
say()

# ---------------------------------------------------------------- 5
say("5. M1 in the MLP arm")
corr13 = (O / "13_mlp_ranking/correlation.txt").read_text()
m = re.search(r"IG vs SHAP:\s*ρ=([+-]?[\d.]+)", corr13)
rho = float(m.group(1))
check("13_mlp_ranking IG vs SHAP rho moved off ~0", abs(rho) > 0.2, f"rho={rho:.3f}")
for l in corr13.strip().splitlines():
    say("    " + l)
say()

# ---------------------------------------------------------------- 6
say("6. Group steering")
g = json.loads((O / "group_steering/group_steering_results.json").read_text())
check("378 rows", len(g) == 378, f"{len(g)} rows")
check("no {probe_type: mlp, method: probe} rows",
      not any(r["probe_type"] == "mlp" and r["method"] == "probe" for r in g))
act = [r for r in g if r["method"] == "actdiff"]
check("actdiff rows all carry arm_independent: true", act and all(r.get("arm_independent") is True for r in act),
      f"{len(act)} rows")
key = lambda r: (r["k"], r["alpha"])
la = {key(r): (r["effect"], r["cost"]) for r in act if r["probe_type"] == "logistic"}
ma = {key(r): (r["effect"], r["cost"]) for r in act if r["probe_type"] == "mlp"}
check("actdiff logistic and mlp copies numerically identical", la == ma and len(la) == 42, f"{len(la)} keys")
say("  counts: " + ", ".join(f"{a}/{b}={n}" for (a, b), n in sorted(Counter((r['probe_type'], r['method']) for r in g).items())))
say()

# ---------------------------------------------------------------- 7
say("7. Steer bookkeeping")
for d in sorted(set(p.parent for p in O.rglob("model_deltas_*.json"))):
    ms = sorted(d.glob("model_deltas_*.json"))
    tags = [p.stem.removeprefix("model_deltas_") for p in ms]
    ok = len(ms) == 1
    detail = f"tags={tags}"
    for t in tags:
        ok &= (d / f"faithfulness_{t}.txt").exists()
        j = json.loads((d / f"model_deltas_{t}.json").read_text())
        gtop = np.load(d / f"shap_top_global_{t}.npy")
        ok &= np.array_equal(gtop, np.array([int(k) for k in j["delta_abs"]]))
    check(f"{rel(d)}", ok, detail)
say()

# ---------------------------------------------------------------- 8
say("8. Determinism (step 2 re-fit into a scratch checkpoint root)")
env = dict(os.environ, SHAP_SAE_OUTPUTS=str(SCRATCH / "outputs"), SHAP_SAE_CHECKPOINTS=str(SCRATCH / "checkpoints"))
for s in ("src/core/2_probes.py", "src/core/3_shap_comp.py"):
    r = subprocess.run(["uv", "run", "python", s], cwd=REPO, env=env, capture_output=True, text=True)
    check(f"re-ran {s} (scratch roots)", r.returncode == 0, r.stderr[-300:] if r.returncode else "")
p0 = joblib.load(C / "probe_layer_7.joblib")
p1 = joblib.load(SCRATCH / "checkpoints/probe_layer_7.joblib")
check("probe coef_ bit-identical", np.array_equal(p0.coef_, p1.coef_) and np.array_equal(p0.intercept_, p1.intercept_))
m_re = np.load(SCRATCH / "outputs/3_shap/shap_feature_indices.npy")
check(f"step-3 mask still ({N_MASK},) and identical", np.array_equal(m_re, m3), f"{m_re.shape}")
say()

# ---------------------------------------------------------------- 9
say("9. Alpha sensitivity (scaled vs constant, steering random k20)")


def parse_faith(p):
    res, block = {}, None
    for l in Path(p).read_text().splitlines():
        if l.startswith("Importance"):
            block = "imp"
        elif l.startswith("Directional"):
            block = "dir"
        m = re.match(r"\s+(\S.*?)\s+rho=([+-][\d.]+)\s+p=([\d.]+)", l)
        if m and block:
            res[(block, m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return res


for arm, d in (("linear", "9_steer"), ("mlp", "14_mlp_steer")):
    s = parse_faith(O / d / "steering_random_k20_scaled/faithfulness_steering_random_k20.txt")
    c = parse_faith(O / d / "steering_random_k20_constant/faithfulness_steering_random_k20.txt")
    say(f"  {arm} arm                       scaled rho (p)     constant rho (p)    gap")
    for k in s:
        say(f"    {k[0]:>3} {k[1]:<22} {s[k][0]:+.3f} ({s[k][1]:.3f})   {c[k][0]:+.3f} ({c[k][1]:.3f})   {s[k][0]-c[k][0]:+.3f}")
    gaps = [abs(s[k][0] - c[k][0]) for k in s]
    say(f"    max |scaled - constant| = {max(gaps):.3f}")
say("  (report only; both reported regardless per CLAUDE.md)")
say()

# ---------------------------------------------------------------- 10
say("10. Causal direction (positive Phi -> positive Delta under steering?)")
phi_lin = np.load(O / "7_shap_recompute/phi_sentiment_layer7_signed.npy")
mlp_idx = np.load(O / "13_mlp_ranking/feature_indices.npy")
phi_mlp = dict(zip(mlp_idx.tolist(), np.load(O / "13_mlp_ranking/shap_scores.npy").tolist()))
for arm, d, phi in (("linear", "9_steer", lambda f: phi_lin[f]), ("mlp", "14_mlp_steer", lambda f: phi_mlp[f])):
    for p in sorted((O / d).glob("steering_*/model_deltas_*.json")):
        j = json.loads(p.read_text())
        f = [int(k) for k in j["delta_signed"]]
        ds = np.array(list(j["delta_signed"].values()))
        ph = np.array([phi(x) for x in f])
        pos, neg = ds[ph > 0], ds[ph < 0]
        fmt = lambda a: f"n={len(a):3d} mean={a.mean():+.4f} agree={np.mean(a > 0) if a is pos else np.mean(a < 0):.2f}" if len(a) else "n=  0"
        say(f"  {arm:6s} {p.parent.name:32s} Phi>0: {fmt(pos)}   Phi<0: {fmt(neg)}")
for name in ("alpha_calibration.txt", "alpha_calibration_logitdiff.txt"):
    t = (O / "9.5_tuning" / name).read_text().splitlines()
    rows = [l.split() for l in t if re.match(r"\s+\d+\.\d+\s+\d", l)]
    say(f"  9.5 pilot ({name}): mean_signed_dp per alpha = " + ", ".join(f"{float(r[0]):.3f}:{r[2]}" for r in rows))
say("  (report only; a reappearing inversion is a finding, not a check failure)")
say()

say(f"SUMMARY: {'ALL HARD CHECKS PASS' if not fails else 'FAILURES: ' + '; '.join(fails)}")
(O / "verification.txt").write_text("\n".join(out) + "\n")
sys.exit(1 if fails else 0)
