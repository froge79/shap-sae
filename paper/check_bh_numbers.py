"""Fail if the BH counts quoted in the paper drift from the sensitivity output.

Run after 9d_bh_family_sensitivity.py, from the repo root:
  SHAP_SAE_OUTPUTS=run3/outputs uv run python paper/check_bh_numbers.py
"""
import json
import os
import sys
from pathlib import Path

root = Path(os.environ.get("SHAP_SAE_OUTPUTS", "run3/outputs"))
j = json.loads((root / "bh_family_sensitivity/bh_family_sensitivity.json").read_text())
tex = "".join(p.read_text() for p in sorted(Path("paper").glob("*.tex")))

def frac(d): return f"{d['bh']}/{d['n_tests']}", f"{d['by']}/{d['n_tests']}"

expected = {
    *frac(j["global_dedup"]["PRIMARY: per run"]),
    *frac(j["global_raw"]["PRIMARY: per run"]),
    *frac(j["paper_wide"]["one family: every inferential test"]),
    *frac(j["paper_wide"]["per run / per local source (primary)"]),
}
missing = sorted(e for e in expected if e not in tex)
flips = j["family_sensitive_tests"]
print("quoted counts:", sorted(expected))
print("family-sensitive tests:", flips)
if missing or flips:
    print("DRIFT: not found in paper:", missing, "| n family-sensitive =", len(flips))
    sys.exit(1)
print("OK")
