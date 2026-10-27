"""
Global DeepSHAP importance as mean |phi| against the FULL train set (MLP probe).

MLP counterpart of `global_linear/7c_global_meanabs.py`: A_j = mean_e |phi_j(x_e)|
over the canonical 1000 eval rows, background = every train row. DeepSHAP is run
with `check_additivity=True` in eval chunks (the full background makes a single
call large); a 100-row-background run must first reproduce the saved
`12_mlp_shap` values exactly.

Output: 7c_global_meanabs/meanabs_mlp.npy (114,), meta_mlp.json
"""
from __future__ import annotations

import importlib.util
import json
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
CHUNK = 125


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    shap12 = _load("mlp_shap12", _SRC / "global_mlp" / "12_mlp_shap.py")
    payload = joblib.load(checkpoint_path(f"mlp_probe_layer_{LAYER}.joblib"))
    probe, fi = payload["probe"], np.asarray(payload["feature_indices"])
    ident = np.arange(len(fi))

    x_eval, bg100, eval_idx, _ = load_eval_and_background(layer=LAYER, n_eval=DEFAULT_N_EVAL)
    x_eval = np.asarray(x_eval[:, fi], dtype=np.float32)

    sv100 = shap12.run_deepshap(probe, x_eval, np.asarray(bg100[:, fi], dtype=np.float32), ident)
    saved = np.load(output_path("12_mlp_shap", "shap_values_raw.npy"))
    err = float(np.abs(sv100 - saved).max())
    assert err < 1e-4, f"bg100 DeepSHAP disagrees with saved values (max err {err:.2e})"

    train = load_activations("train", LAYER, "activations", mmap=True)
    n_train = train.shape[0]
    _, order = sample_eval_rows(layer=LAYER, n_eval=DEFAULT_N_EVAL, n_background=n_train)
    bg = np.asarray(train[order][:, fi], dtype=np.float32)

    total = np.zeros(len(fi))
    for s in range(0, len(x_eval), CHUNK):
        sv = shap12.run_deepshap(probe, x_eval[s : s + CHUNK], bg, ident)
        total += np.abs(sv).sum(0)
        print(f"  eval rows {s + len(sv)}/{len(x_eval)}", flush=True)
    score = total / len(x_eval)

    out = output_path("7c_global_meanabs")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "meanabs_mlp.npy", score)
    json.dump(
        {"n_eval": len(eval_idx), "n_background": int(n_train), "bg100_max_err": err},
        open(out / "meta_mlp.json", "w"),
        indent=2,
    )
    print(f"wrote {out}/meanabs_mlp.npy  shape={score.shape}  bg={n_train}")


if __name__ == "__main__":
    main()
