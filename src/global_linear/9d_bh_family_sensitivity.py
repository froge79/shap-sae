"""
Multiple-comparison sensitivity for the paper's inferential tests.

Reanalysis only (CPU, seconds). Reads raw p-values from
  * each `--selection all` run's faithfulness_<tag>_bh.json sidecar (global), and
  * each local run's paired_tests.json (crossed-bootstrap p; the Wilcoxon p is
    examples-only and not quoted),
and recounts BH survivors under alternative test families.

Family rule (fixed by the analysis design, not by the results): the primary
family is the tests of ONE `all` run. Everything else here is a sensitivity row.

Two corrections to the naive count:
  * Dedup. In the linear arm SHAP, IG and GA have identical (rho, p) by
    construction, so they are one test. A test whose (rho, p) exactly equals an
    already-kept test in the same run is dropped; the "raw" rows keep them.
  * Dependence. BH assumes independence or positive dependence. The scope
    pairs share features and eval rows, so every row is also run under
    Benjamini-Yekutieli, which is valid under arbitrary dependence.

Output: bh_family_sensitivity.{json,txt} with (a) survivor counts per family
definition and (b) a per-test table giving q under the narrow (primary) family
and under the paper-wide family, flagging any test that survives only the narrow
one.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import false_discovery_control

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import output_path  # noqa: E402

ALPHA = 0.05

GLOBAL_RUNS = {  # (arm, scope) -> directory under outputs
    ("linear", "all-token"): "9_steer/steering_all_scaled",
    ("linear", "last-token"): "9_steer/steering_all_scaled_lasttoken",
    ("mlp", "all-token"): "14_mlp_steer/steering_all_scaled",
    ("mlp", "last-token"): "14_mlp_steer/steering_all_scaled_lasttoken",
}
LOCAL_RUNS = {  # source -> directory under outputs
    "linear": "9_local_steer",
    "mlp": "14_local_mlp_steer",
    "lm-diag": "local_shap_faithfulness",
}


def adjust(ps: dict[str, float], method: str) -> dict[str, tuple[float, bool]]:
    """{name: (q, reject)}; NaN p held out."""
    names = list(ps)
    raw = np.asarray([ps[n] for n in names], dtype=float)
    ok = np.isfinite(raw)
    q = np.full_like(raw, np.nan)
    if ok.any():
        q[ok] = false_discovery_control(raw[ok], method=method)
    return {n: (float(q[i]), bool(ok[i] and q[i] <= ALPHA)) for i, n in enumerate(names)}


def load_global(dedupe: bool) -> dict[str, float]:
    """name -> p, name = global|arm|scope|test."""
    out: dict[str, float] = {}
    for (arm, scope), d in GLOBAL_RUNS.items():
        f = next(output_path(d).glob("faithfulness_*_bh.json"))
        seen: set[tuple[float, float]] = set()
        for test, v in json.loads(f.read_text())["tests"].items():
            sig = (round(v["rho"], 12), round(v["p"], 15))
            if dedupe and sig in seen:
                continue
            seen.add(sig)
            out[f"global|{arm}|{scope}|{test}"] = v["p"]
    return out


def load_local() -> dict[str, float]:
    out: dict[str, float] = {}
    for src, d in LOCAL_RUNS.items():
        blocks = json.loads((output_path(d) / "paired_tests.json").read_text())["blocks"]
        for blk, v in blocks.items():
            out[f"local|{src}|{blk}"] = v["crossed"]["p_two_sided"]
    return out


def subset(ps: dict[str, float], pred) -> dict[str, float]:
    return {k: v for k, v in ps.items() if pred(k.split("|"))}


def count(fams: list[dict[str, float]], method: str) -> tuple[int, int, set[str]]:
    res = {k: v for f in fams for k, v in adjust(f, method).items()}
    surv = {k for k, (_, r) in res.items() if r}
    return len(surv), len(res), surv


def main() -> None:
    g_raw, g_ded, loc = load_global(False), load_global(True), load_local()
    lines: list[str] = []
    out: dict = {"alpha": ALPHA}

    def section(title: str, defs: dict[str, list[dict[str, float]]], key: str) -> None:
        lines.append(f"\n{title}")
        lines.append(f"{'family definition':<44}{'BH':>10}{'BY':>10}")
        out[key] = {}
        for name, fams in defs.items():
            nb, n, sb = count(fams, "bh")
            ny, _, sy = count(fams, "by")
            lines.append(f"{name:<44}{f'{nb}/{n}':>10}{f'{ny}/{n}':>10}")
            out[key][name] = {"n_tests": n, "bh": nb, "by": ny,
                              "bh_survivors": sorted(sb), "by_survivors": sorted(sy)}

    def global_defs(g: dict[str, float]) -> dict[str, list[dict[str, float]]]:
        runs = [subset(g, lambda p, a=a, s=s: (p[1], p[2]) == (a, s)) for (a, s) in GLOBAL_RUNS]
        return {
            "PRIMARY: per run": runs,
            "pooled over arms, per scope": [subset(g, lambda p, s=s: p[2] == s)
                                            for s in ("all-token", "last-token")],
            "pooled over scopes, per arm": [subset(g, lambda p, a=a: p[1] == a)
                                            for a in ("linear", "mlp")],
            "pooled over all global runs": [g],
        }

    section("GLOBAL, raw (SHAP/IG/GA counted separately in linear arm)",
            global_defs(g_raw), "global_raw")
    section("GLOBAL, deduplicated (identical (rho, p) counted once)",
            global_defs(g_ded), "global_dedup")

    paper = {**g_ded, **loc}
    section("PAPER-WIDE (deduplicated global + local crossed tests)", {
        "per run / per local source (primary)": [
            *[subset(g_ded, lambda p, a=a, s=s: (p[1], p[2]) == (a, s)) for (a, s) in GLOBAL_RUNS],
            *[subset(loc, lambda p, s=s: p[1] == s) for s in LOCAL_RUNS]],
        "global pooled; local pooled": [g_ded, loc],
        "one family: every inferential test": [paper],
    }, "paper_wide")

    # Per-test: narrow (primary) vs paper-wide q.
    narrow = {}
    for (a, s) in GLOBAL_RUNS:
        narrow.update(adjust(subset(g_ded, lambda p, a=a, s=s: (p[1], p[2]) == (a, s)), "bh"))
    for src in LOCAL_RUNS:
        narrow.update(adjust(subset(loc, lambda p, src=src: p[1] == src), "bh"))
    wide_bh, wide_by = adjust(paper, "bh"), adjust(paper, "by")
    lines.append("\nPER TEST (q under primary family vs one paper-wide family)")
    lines.append(f"{'test':<46}{'p':>10}{'q_narrow':>10}{'q_wide_BH':>10}{'q_wide_BY':>10}  flag")
    rows, flips = [], []
    for k in paper:
        (qn, rn), (qb, rb), (qy, ry) = narrow[k], wide_bh[k], wide_by[k]
        flag = "narrow-only" if rn and not rb else ("BY-fails" if rb and not ry else "")
        if flag:
            flips.append(k)
        rows.append({"test": k, "p": paper[k], "q_narrow": qn, "q_wide_bh": qb,
                     "q_wide_by": qy, "flag": flag})
        lines.append(f"{k:<46}{paper[k]:>10.2g}{qn:>10.3g}{qb:>10.3g}{qy:>10.3g}  {flag}")
    out["per_test"] = rows
    out["family_sensitive_tests"] = flips
    lines.append(f"\nTests whose verdict depends on the family: {len(flips)}"
                 + (f" ({', '.join(flips)})" if flips else ""))
    lines.append("Local p-values are crossed-bootstrap (B=1000, resolution 0.001); "
                 "sign tests and minimal-pair summaries carry no inferential claim "
                 "in the paper and are excluded.")

    d = output_path("bh_family_sensitivity")
    d.mkdir(parents=True, exist_ok=True)
    (d / "bh_family_sensitivity.json").write_text(json.dumps(out, indent=2))
    (d / "bh_family_sensitivity.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
