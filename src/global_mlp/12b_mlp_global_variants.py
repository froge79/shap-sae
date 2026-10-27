"""
MLP-arm counterpart of `global_linear/7b_global_variants.py`: DeepSHAP global
scores under mean|phi| vs mean-signed, with larger and bootstrapped backgrounds,
rescored against the saved `14_mlp_steer` deltas (no new LM forward passes).

Unlike the linear probe there is no closed form, so each background is a real
DeepSHAP pass (`check_additivity=True`, as in 12_mlp_shap). `bg100` is rerun and
must reproduce the saved Phi — that is the guard that this script explains the
same function on the same rows.

`mean_abs` is unsigned, so it has no directional entry; the MLP has no signed
weight to lend it a direction.

Output: 12b_mlp_global_variants/global_variants.{json,txt}
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

_SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_SRC))
from utils import (  # noqa: E402
    CANONICAL_SEED,
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
    spec.loader.exec_module(mod)
    return mod


_shap12 = _load("mlp_shap12", _SRC / "global_mlp" / "12_mlp_shap.py")
_lin = _load("global_variants7b", _SRC / "global_linear" / "7b_global_variants.py")

RUNS = [
    (lab, d.replace("9_steer", "14_mlp_steer"), t) for lab, d, t in _lin.RUNS
]


def scores(shap_values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "signed_mean": shap_values.mean(0),
        "mean_abs": np.abs(shap_values).mean(0),
    }


def main(
    n_boot: int, boot_sizes: list[int], large_sizes: list[int], full: bool, seed: int, out_dir: Path
):
    payload = joblib.load(checkpoint_path(f"mlp_probe_layer_{LAYER}.joblib"))
    probe = payload["probe"]
    fi = np.asarray(payload["feature_indices"])

    x_eval, bg100, _, _ = load_eval_and_background(layer=LAYER, n_eval=DEFAULT_N_EVAL)
    train = load_activations("train", LAYER, "activations", mmap=True)
    n_train = train.shape[0]
    _, order = sample_eval_rows(layer=LAYER, n_eval=DEFAULT_N_EVAL, n_background=n_train)
    x_train = np.asarray(train[order][:, fi], dtype=np.float32)
    x_eval_f = np.asarray(x_eval[:, fi], dtype=np.float32)

    def deepshap(bg_rows: np.ndarray) -> np.ndarray:
        # Inputs are already restricted to the pool, so the column index is a no-op.
        return _shap12.run_deepshap(probe, x_eval_f, bg_rows, np.arange(len(fi)))

    # Guard: bg100 must reproduce the saved attribution.
    sv100 = deepshap(np.asarray(bg100[:, fi], dtype=np.float32))
    saved = np.load(output_path("12_mlp_shap", "shap_values_raw.npy"))
    err = float(np.abs(sv100 - saved).max())
    assert err < 1e-4, f"bg100 disagrees with saved DeepSHAP (max err {err:.2e})"

    runs = {lab: _lin.load_deltas(d, t) for lab, d, t in RUNS}
    pos = {int(f): i for i, f in enumerate(fi)}

    def run_rhos(s: np.ndarray, directional: bool):
        out = {}
        for lab, (feats, dabs, dsig) in runs.items():
            v = s[[pos[int(f)] for f in feats]]
            out[lab] = {"importance": _lin.rho(v, dabs, dsig, "importance")}
            if directional:
                out[lab]["directional"] = _lin.rho(v, dabs, dsig, "directional")
        return out

    res = {"closed_check_max_err": err, "n_pool": int(len(fi)), "point": {}, "boot": {}}

    def add_point(tag: str, sv: np.ndarray):
        for sn, s in scores(sv).items():
            res["point"][f"{sn}@{tag}"] = run_rhos(s, sn == "signed_mean")

    add_point("bg100", sv100)
    for n in large_sizes + ([n_train] if full else []):
        add_point("bgfull" if n == n_train else f"bg{n}", deepshap(x_train[:n]))
        print(f"done point bg={n}", flush=True)

    rng = np.random.default_rng(seed)
    for n in boot_sizes:
        draws: dict[str, dict] = {}
        for b in range(n_boot):
            sv = deepshap(x_train[rng.integers(0, n_train, size=n)])
            for sn, s in scores(sv).items():
                for lab, r in run_rhos(s, sn == "signed_mean").items():
                    for kind, v in r.items():
                        draws.setdefault(f"{sn}@boot{n}", {}).setdefault(
                            f"{lab}|{kind}", []
                        ).append(v)
        for key, d in draws.items():
            res["boot"][key] = {
                k: {
                    "mean": float(np.mean(v)),
                    "lo": float(np.percentile(v, 2.5)),
                    "hi": float(np.percentile(v, 97.5)),
                    "frac_pos": float(np.mean(np.asarray(v) > 0)),
                }
                for k, v in d.items()
            }
        print(f"done bootstrap size={n}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out_dir / "global_variants.json", "w"), indent=2)
    L = [f"MLP arm global variants — pool n={res['n_pool']}, bootstrap B={n_boot} "
         f"(sizes {boot_sizes}); bg100 reproduces saved DeepSHAP (max err {err:.1e})"]
    for kind in ("importance", "directional"):
        L += ["", f"== {kind} faithfulness (Spearman rho vs causal effect) =="]
        for lab in runs:
            L.append(f"-- {lab}")
            for key, r in res["point"].items():
                if kind in r[lab]:
                    L.append(f"   {key:22s} {r[lab][kind]:+.3f}")
            for key, r in res["boot"].items():
                b = r.get(f"{lab}|{kind}")
                if b:
                    L.append(f"   {key:22s} {b['mean']:+.3f}  [{b['lo']:+.3f}, {b['hi']:+.3f}]  "
                             f"frac>0 {b['frac_pos']:.2f}")
    text = "\n".join(L) + "\n"
    (out_dir / "global_variants.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=30)
    ap.add_argument("--boot-sizes", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--large-sizes", type=int, nargs="+", default=[1000])
    ap.add_argument("--full", action="store_true", help="also run DeepSHAP against every train row (slow)")
    ap.add_argument("--seed", type=int, default=CANONICAL_SEED)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    main(a.n_boot, a.boot_sizes, a.large_sizes, a.full, a.seed,
         a.out_dir or output_path("12b_mlp_global_variants"))
