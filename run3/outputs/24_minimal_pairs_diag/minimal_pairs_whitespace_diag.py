"""
Diagnostic: does the edited text's trailing space (absent from the sentence-level
originals) drive 24_minimal_pairs' readout sign accuracy? Same pairs (same seed and
builder calls as main()), same SAE-recon logit-diff scoring; only trailing
whitespace varies. Writes run3/outputs/24_minimal_pairs_diag/.
"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
from sae_lens import SAE
sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("mp", "src/groundtruth/24_minimal_pairs.py")
mp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mp)
from utils import load_model, load_splits, output_path

out = output_path("24_minimal_pairs_diag"); out.mkdir(parents=True, exist_ok=True)
_, _, shap = load_splits(); sents = list(shap["sentence"])
model = load_model()
sae = SAE.from_pretrained(mp.SAE_RELEASE, mp.SAE_ID).to(model.cfg.device)
pos_id, neg_id = mp._steer9.sentiment_token_ids(model)
hp = f"blocks.{mp.LAYER}.hook_resid_post"
recon = lambda r, hook: sae.decode(sae.encode(r))
score = lambda t: mp.score_with_hooks(model, model.to_tokens([t]), pos_id, neg_id, [(hp, recon)])

variants = {
    "as_built (orig no space, edited + space)": lambda o, e: (o, e),
    "both stripped": lambda o, e: (o.rstrip(), e.rstrip()),
    "both + trailing space": lambda o, e: (o.rstrip() + " ", e.rstrip() + " "),
}
res, lines = {}, ["Minimal-pair readout sign accuracy vs trailing whitespace (n=100, seed=42)", ""]
for edit in ("remove", "insert"):
    rng = np.random.default_rng(42)
    builder = mp.build_remove_pairs if edit == "remove" else mp.build_insert_pairs
    pairs = builder(sents, rng, 100)
    for name, f in variants.items():
        acc = []
        for p in pairs:
            o, e = f(p["original"], p["edited"])
            acc.append(np.sign(score(e) - score(o)) == p["expected_sign"])
        res[f"{edit} | {name}"] = float(np.mean(acc))
        lines.append(f"  {edit:6s}  {name:42s} sign_acc = {np.mean(acc):.3f}")
        print(lines[-1], flush=True)
(out / "whitespace_sign_accuracy.json").write_text(json.dumps(res, indent=2))
(out / "whitespace_sign_accuracy.txt").write_text("\n".join(lines) + "\n")
