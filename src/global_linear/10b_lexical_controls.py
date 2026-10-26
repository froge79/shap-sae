"""
Step 18b — stronger lexical controls (answers "the BoW baseline is a hand
lexicon" and "last-token vs lexicon-anywhere is a weak word-detector test").

10_simplicity_check.py's "bag-of-words" baseline is a logistic regression on
counts of ~100 hand-picked lexicon words, and its lexical-tracking test is a
point-biserial r between a last-token activation and "any lexicon word anywhere
in the sentence". This script keeps both as reference rows and adds:

  Part 1 — baselines (fit on train, C tuned on VAL only, scored on held-out once)
    * lexicon-count LR (the old baseline, for continuity)
    * full-vocabulary binary-unigram BoW LR
    * unigram+bigram BoW LR, and TF-IDF unigram+bigram LR
    * dense layer-7 last-token residual LR (same representation as the SAE
      probe, without the SAE bottleneck / sparsity)
    * the SAE probe itself, with Wilson CI, and McNemar + paired-bootstrap tests
      of (probe - baseline)

  Part 2 — are features / the probe word detectors?
    * held-out R^2 of ridge(full BoW -> activation) for the top-20 |Phi|
      features vs the remaining masked features, and vs lexicon-only ridge
    * held-out R^2 of ridge(BoW -> probe logit): is the probe reducible to
      word-spotting?
    * signed and position-aware lexical correlations: r with has_pos / has_neg,
      sign(Phi) * Spearman(activation, net lexicon score), and point-biserial r
      with "lexicon word in the last 3 tokens"

READING RULES — fixed before the first run (do not move them afterwards):
  R1  The old rule stands: "BoW near probe" if best-val BoW held-out accuracy
      >= probe accuracy - 0.03. If so, the claim that non-lexical signal
      remains is dropped, not hedged.
  R2  Probe is "mostly word-spotting" if held-out R^2(BoW -> probe logit) >= 0.50,
      "not reducible" if < 0.25, otherwise "partly".
  R3  A top feature is "BoW-explained" if held-out R^2(BoW -> activation) >= 0.25.
      Report the count out of 20 and the comparison with the masked-feature
      distribution; no further threshold.
  R4  Nothing is tuned on held-out. Hyperparameters come from val.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.stats import binomtest, pointbiserialr, spearmanr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    CANONICAL_SEED,
    checkpoint_path,
    last_real_token_index,
    load_model,
    load_splits,
    output_path,
)

LAYER = 7
N_BOOT = 2000


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sim = _load("simplicity10", Path(__file__).resolve().parent / "10_simplicity_check.py")
LEX = sorted(sim.POS_WORDS | sim.NEG_WORDS)


def an_uni(s: str) -> list[str]:
    return sim.tokenize(s)


def an_bi(s: str) -> list[str]:
    t = sim.tokenize(s)
    return t + [f"{a}_{b}" for a, b in zip(t[:-1], t[1:])]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def residuals(model, sentences, batch_size=32):
    hook = f"blocks.{LAYER}.hook_resid_post"
    out = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            tok = model.to_tokens(sentences[i : i + batch_size])
            _, cache = model.run_with_cache(tok, names_filter=hook)
            last = last_real_token_index(model, tok)
            idx = torch.arange(tok.shape[0], device=tok.device)
            out.append(cache[hook][idx, last, :].float().cpu().numpy())
    return np.concatenate(out)


def fit_lr_tuned(Xtr, ytr, Xva, yva, Xte, yte, Cs):
    best = None
    for C in Cs:
        clf = LogisticRegression(C=C, max_iter=3000, random_state=CANONICAL_SEED)
        clf.fit(Xtr, ytr)
        va = accuracy_score(yva, clf.predict(Xva))
        if best is None or va > best[0]:
            best = (va, C, clf)
    va, C, clf = best
    pred = clf.predict(Xte)
    return {
        "C": C,
        "val_acc": float(va),
        "test_acc": float(accuracy_score(yte, pred)),
        "test_auroc": float(roc_auc_score(yte, clf.decision_function(Xte))),
        "pred": pred,
    }


def ridge_r2(Xtr, ytr, Xva, yva, Xte, yte, alphas=(0.1, 1, 10, 100, 1000)):
    """Ridge with alpha chosen on val R^2; returns held-out R^2. ytr may be (n,) or (n,k)."""
    best = None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        v = r2_score(yva, m.predict(Xva))
        if best is None or v > best[0]:
            best = (v, a, m)
    return float(r2_score(yte, best[2].predict(Xte))), best[1]


def main():
    out_dir = output_path("10b_lexical_controls")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, te_ds = load_splits()
    S = {"tr": list(train_ds["sentence"]), "va": list(val_ds["sentence"]), "te": list(te_ds["sentence"])}
    Y = {"tr": np.asarray(train_ds["label"]), "va": np.asarray(val_ds["label"]), "te": np.asarray(te_ds["label"])}

    probe = joblib.load(checkpoint_path("probe_layer_7.joblib"))
    probe.verbose = 0
    A = {
        "tr": np.load("activations/probe_train/layer_7/activations.npy", mmap_mode="r"),
        "va": np.load("activations/probe_val/layer_7/activations.npy", mmap_mode="r"),
        "te": np.load("activations/shap/layer_7/activations.npy", mmap_mode="r"),
    }
    for k in A:
        assert A[k].shape[0] == len(S[k])

    def logits(a, bs=512):
        return np.concatenate(
            [probe.decision_function(np.asarray(a[i : i + bs], dtype=np.float32)) for i in range(0, a.shape[0], bs)]
        )

    L = {k: logits(A[k]) for k in A}
    probe_pred = (L["te"] > 0).astype(int)
    probe_acc = float(accuracy_score(Y["te"], probe_pred))
    res = {"n_test": len(S["te"]), "probe": {
        "test_acc": probe_acc, "wilson95": wilson(int((probe_pred == Y["te"]).sum()), len(Y["te"])),
        "val_acc": float(accuracy_score(Y["va"], (L["va"] > 0).astype(int))),
        "test_auroc": float(roc_auc_score(Y["te"], L["te"]))}}

    # ---------------- Part 1: baselines ----------------
    Cs = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)
    vecs = {
        "lexicon_count": CountVectorizer(analyzer=an_uni, vocabulary=LEX),
        "bow_unigram_binary": CountVectorizer(analyzer=an_uni, binary=True),
        "bow_uni_bigram_binary": CountVectorizer(analyzer=an_bi, binary=True),
        "tfidf_uni_bigram": TfidfVectorizer(analyzer=an_bi, sublinear_tf=True),
    }
    base = {}
    preds = {}
    for name, v in vecs.items():
        Xtr = v.fit_transform(S["tr"])
        Xva, Xte = v.transform(S["va"]), v.transform(S["te"])
        r = fit_lr_tuned(Xtr, Y["tr"], Xva, Y["va"], Xte, Y["te"], Cs)
        r["vocab"] = Xtr.shape[1]
        preds[name] = r.pop("pred")
        base[name] = r

    print("loading GPT-2 for the dense layer-7 comparator ...")
    model = load_model()
    R = {k: residuals(model, S[k]) for k in S}
    del model
    torch.cuda.empty_cache()
    sc = StandardScaler().fit(R["tr"])
    r = fit_lr_tuned(sc.transform(R["tr"]), Y["tr"], sc.transform(R["va"]), Y["va"],
                     sc.transform(R["te"]), Y["te"], (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1))
    r["vocab"] = 768
    preds["dense_residual_last_token"] = r.pop("pred")
    base["dense_residual_last_token"] = r

    brng = np.random.default_rng(0)
    n = len(Y["te"])
    boots = brng.integers(0, n, size=(N_BOOT, n))
    ok_probe = (probe_pred == Y["te"]).astype(float)
    for name, p in preds.items():
        ok_b = (p == Y["te"]).astype(float)
        base[name]["wilson95"] = wilson(int(ok_b.sum()), n)
        d = ok_probe - ok_b
        boot = d[boots].mean(axis=1)
        b01 = int(((ok_probe == 1) & (ok_b == 0)).sum())
        b10 = int(((ok_probe == 0) & (ok_b == 1)).sum())
        base[name]["probe_minus_baseline"] = {
            "diff": float(d.mean()),
            "boot95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "mcnemar_p": float(binomtest(b01, b01 + b10, 0.5).pvalue) if b01 + b10 else 1.0,
            "probe_only_right": b01, "baseline_only_right": b10,
        }
    res["baselines"] = base
    best_bow = max((k for k in base if k.startswith(("bow", "tfidf"))), key=lambda k: base[k]["val_acc"])
    res["best_bow_by_val"] = best_bow
    res["R1_bow_near_probe"] = bool(base[best_bow]["test_acc"] >= probe_acc - 0.03)

    # ---------------- Part 2: word-detector tests ----------------
    feature_indices = np.load(output_path("3_shap", "shap_feature_indices.npy"))
    phi = np.load(output_path("7_shap_recompute", "phi_sentiment_layer7_signed.npy"))
    ranked = [int(f) for f in np.argsort(np.abs(phi))[::-1] if int(f) in set(map(int, feature_indices))]
    top = ranked[:20]
    rest = [int(f) for f in feature_indices if int(f) not in set(top)]
    cols = top + rest
    Fa = {k: np.asarray(A[k][:, cols], dtype=np.float64) for k in A}

    vb = CountVectorizer(analyzer=an_uni, binary=True)
    Btr = vb.fit_transform(S["tr"]); Bva = vb.transform(S["va"]); Bte = vb.transform(S["te"])
    vl = CountVectorizer(analyzer=an_uni, vocabulary=LEX)
    Ltr = vl.fit_transform(S["tr"]); Lva = vl.transform(S["va"]); Lte = vl.transform(S["te"])

    def per_feature_r2(Xtr, Xva, Xte):
        out = []
        for j in range(len(cols)):
            r2, _ = ridge_r2(Xtr, Fa["tr"][:, j], Xva, Fa["va"][:, j], Xte, Fa["te"][:, j])
            out.append(r2)
        return np.array(out)

    r2_bow = per_feature_r2(Btr, Bva, Bte)
    r2_lex = per_feature_r2(Ltr, Lva, Lte)
    res["feature_r2"] = {
        "top20_bow": r2_bow[:20].tolist(), "top20_lexicon": r2_lex[:20].tolist(),
        "top20_features": top,
        "top20_median_bow": float(np.median(r2_bow[:20])),
        "top20_median_lexicon": float(np.median(r2_lex[:20])),
        "rest_median_bow": float(np.median(r2_bow[20:])),
        "rest_q90_bow": float(np.percentile(r2_bow[20:], 90)),
        "rest_median_lexicon": float(np.median(r2_lex[20:])),
        "n_rest": len(rest),
        "R3_n_top20_bow_explained_ge_0.25": int((r2_bow[:20] >= 0.25).sum()),
    }
    pr_bow, a1 = ridge_r2(Btr, L["tr"], Bva, L["va"], Bte, L["te"])
    pr_lex, a2 = ridge_r2(Ltr, L["tr"], Lva, L["va"], Lte, L["te"])
    res["probe_logit_reducibility"] = {
        "r2_full_bow": pr_bow, "r2_lexicon_only": pr_lex,
        "R2_reading": "mostly word-spotting" if pr_bow >= 0.5 else ("not reducible" if pr_bow < 0.25 else "partly"),
    }

    # Signed and position-aware correlations on held-out, top-20.
    flags = sim.lexicon_flags(S["te"])
    last3 = np.array([bool(set(sim.tokenize(s)[-3:]) & (sim.POS_WORDS | sim.NEG_WORDS)) for s in S["te"]])
    rows = []
    for j, f in enumerate(top):
        a = Fa["te"][:, j]
        rho, _ = spearmanr(a, flags["lex_score"])
        row = {
            "feature": f, "phi": float(phi[f]),
            "r_has_pos": float(pointbiserialr(flags["has_pos"].astype(float), a)[0]),
            "r_has_neg": float(pointbiserialr(flags["has_neg"].astype(float), a)[0]),
            "r_last3": float(pointbiserialr(last3.astype(float), a)[0]),
            "rho_net_lex": float(rho),
        }
        rows.append(row)
    res["signed_position_aware"] = {
        "frac_sentences_lexicon_in_last3": float(last3.mean()),
        "mean_abs_r_has_pos": float(np.nanmean([abs(r["r_has_pos"]) for r in rows])),
        "mean_abs_r_has_neg": float(np.nanmean([abs(r["r_has_neg"]) for r in rows])),
        "mean_abs_r_last3": float(np.nanmean([abs(r["r_last3"]) for r in rows])),
        "mean_abs_rho_net_lex": float(np.nanmean([abs(r["rho_net_lex"]) for r in rows])),
        "mean_signed_agreement_sign_phi_x_rho_net_lex": float(
            np.nanmean([np.sign(r["phi"]) * r["rho_net_lex"] for r in rows])
        ),
        "per_feature": rows,
    }

    (out_dir / "lexical_controls.json").write_text(json.dumps(res, indent=2))

    f3 = lambda x: f"{x:.3f}"
    T = ["Lexical controls (held-out n=%d; fit on train, C/alpha tuned on val)" % res["n_test"], "=" * 74,
         f"SAE probe            acc {f3(probe_acc)}  Wilson95 {res['probe']['wilson95']}  AUROC {f3(res['probe']['test_auroc'])}",
         "", f"{'baseline':28s} {'vocab':>6s} {'val':>6s} {'test':>6s} {'AUROC':>6s}  probe-minus-baseline [boot95]  McNemar p"]
    for k, v in base.items():
        d = v["probe_minus_baseline"]
        T.append(f"{k:28s} {v['vocab']:6d} {f3(v['val_acc'])} {f3(v['test_acc'])} {f3(v['test_auroc'])}  "
                 f"{d['diff']:+.3f} [{d['boot95'][0]:+.3f},{d['boot95'][1]:+.3f}]  {d['mcnemar_p']:.3g}")
    T += ["", f"best BoW by val: {best_bow}; R1 (BoW within 0.03 of probe): {res['R1_bow_near_probe']}", ""]
    fr = res["feature_r2"]
    T += ["Held-out R^2, ridge -> feature activation",
          f"  top-20 |Phi|   median full-BoW {f3(fr['top20_median_bow'])}   lexicon-only {f3(fr['top20_median_lexicon'])}",
          f"  other masked ({fr['n_rest']})  median full-BoW {f3(fr['rest_median_bow'])}  q90 {f3(fr['rest_q90_bow'])}   lexicon-only {f3(fr['rest_median_lexicon'])}",
          f"  R3: top-20 with BoW R^2 >= 0.25: {fr['R3_n_top20_bow_explained_ge_0.25']}/20", ""]
    pr = res["probe_logit_reducibility"]
    T += [f"Probe logit reducibility: R^2 full BoW {f3(pr['r2_full_bow'])}, lexicon-only {f3(pr['r2_lexicon_only'])}  -> {pr['R2_reading']}", ""]
    sp = res["signed_position_aware"]
    T += ["Top-20 signed / position-aware lexical correlations (held-out)",
          f"  mean|r| has_pos {f3(sp['mean_abs_r_has_pos'])}  has_neg {f3(sp['mean_abs_r_has_neg'])}  "
          f"lexicon-in-last-3 {f3(sp['mean_abs_r_last3'])} (present in {sp['frac_sentences_lexicon_in_last3']:.2f} of sentences)",
          f"  mean|rho| net lexicon score {f3(sp['mean_abs_rho_net_lex'])};  mean sign(Phi)*rho {sp['mean_signed_agreement_sign_phi_x_rho_net_lex']:+.3f}"]
    (out_dir / "lexical_controls.txt").write_text("\n".join(T))
    print("\n".join(T))


if __name__ == "__main__":
    main()
