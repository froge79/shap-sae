"""
Step 0: build the sentence-level three-way split. CPU only, ~seconds.

Why this exists: the earlier split was a random row split of the 67,349-row
phrase-level SST-2 train set, so sub-phrases of one review landed in different
splits (83.8% of held-out phrases were contiguous sub-phrases of a train phrase).
This script uses the sentence-level SST-2 release (SetFit/sst2: full sentences,
6920 train / 872 validation / 1821 test, all labelled), pools the three pieces,
drops exact duplicate texts, and re-splits 70/15/15.

Writes:
  data/sst2_sentences/             HF dataset, columns: sentence, label
  data/three_way_split_indices.json  {train,val,shap}_indices into that dataset

Then runs a leakage gate and exits non-zero if any text is shared across splits.
Split indices are positions in `data/sst2_sentences`, and every extracted
activation is in this row order, so do not re-run this after step 1 without
deleting `activations/`.

    uv run python src/core/0_build_split.py --force    # refuses without --force if a split exists
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_dataset

SOURCE = "SetFit/sst2"
DATA_DIR = Path("data/sst2_sentences")
INDEX_PATH = Path("data/three_way_split_indices.json")
SEED = 42


def load_pooled() -> Dataset:
    raw = load_dataset(SOURCE)
    parts = []
    for name in ("train", "validation", "test"):
        d = raw[name]
        parts.append(
            Dataset.from_dict({"sentence": [t.strip() for t in d["text"]], "label": d["label"]})
        )
    pooled = concatenate_datasets(parts)
    print(f"pooled {len(pooled)} rows from {SOURCE}")
    return pooled


def dedupe(ds: Dataset) -> Dataset:
    """Drop repeated texts (keep first). A text with two labels would be ambiguous, drop all of it."""
    seen: dict[str, int] = {}
    conflicted: set[str] = set()
    for s, y in zip(ds["sentence"], ds["label"]):
        if s in seen and seen[s] != y:
            conflicted.add(s)
        seen.setdefault(s, y)
    keep, taken = [], set()
    for i, s in enumerate(ds["sentence"]):
        if s in conflicted or s in taken:
            continue
        taken.add(s)
        keep.append(i)
    print(f"dedupe: {len(ds)} -> {len(keep)} rows ({len(conflicted)} label-conflicted texts dropped)")
    return drop_contained(ds.select(keep))


def drop_contained(ds: Dataset) -> Dataset:
    """
    Drop any text that appears, on word boundaries, inside a different text.

    Residual phrase-level overlap even in the sentence data: a one-word row like
    'terrible .' is also a fragment of 'truly terrible .'. Dropped from the pool
    so it cannot straddle splits in either direction.
    """
    S = ds["sentence"]
    blob = "\n".join(f" {t} " for t in S)
    keep = []
    for i, t in enumerate(S):
        # the text occurs once as itself; a second occurrence means another row contains it
        if blob.count(f" {t} ") > 1:
            continue
        keep.append(i)
    print(f"drop_contained: {len(S)} -> {len(keep)} rows")
    return ds.select(keep)


def leakage_gate(ds: Dataset, idx: dict[str, list[int]]) -> bool:
    S = ds["sentence"]
    tr = [S[i] for i in idx["train_indices"]]
    ok = True
    train_set = set(tr)
    # padded so a match respects word boundaries
    train_blob = "\n".join(f" {t} " for t in tr)
    for name in ("val_indices", "shap_indices"):
        texts = [S[i] for i in idx[name]]
        dup = sum(t in train_set for t in texts)
        contained = sum(f" {t} " in train_blob for t in texts)
        print(f"  {name}: exact dup in train = {dup}, contained in a train text = {contained} / {len(texts)}")
        if dup or contained:
            ok = False
    a, b, c = (set(idx[k]) for k in ("train_indices", "val_indices", "shap_indices"))
    disjoint = not (a & b or a & c or b & c)
    print(f"  index sets disjoint: {disjoint}")
    return ok and disjoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if (INDEX_PATH.exists() or DATA_DIR.exists()) and not args.force:
        print(f"{INDEX_PATH} or {DATA_DIR} already exists; pass --force to overwrite.")
        return 1
    if args.force and Path("activations").exists():
        print("refusing: activations/ exists and would be decoupled from a new split. Move it aside first.")
        return 1

    ds = dedupe(load_pooled())
    lens = [len(s.split()) for s in ds["sentence"]]
    print(f"median words {statistics.median(lens)}, <=3 words {sum(l <= 3 for l in lens) / len(lens):.3%}")

    # Same two-stage recipe as before (seed 42, 70 / 15 / 15), on sentences.
    ds_id = ds.add_column("original_index", range(len(ds)))
    first = ds_id.train_test_split(test_size=0.30, seed=SEED)
    second = first["test"].train_test_split(test_size=0.5, seed=SEED)
    idx = {
        "train_indices": [int(i) for i in first["train"]["original_index"]],
        "val_indices": [int(i) for i in second["train"]["original_index"]],
        "shap_indices": [int(i) for i in second["test"]["original_index"]],
    }

    print("leakage gate:")
    if not leakage_gate(ds, idx):
        print("FAIL: text is shared across splits; nothing written.")
        return 1

    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(DATA_DIR))
    with open(INDEX_PATH, "w") as f:
        json.dump(idx, f, indent=4)
    pos = sum(ds["label"]) / len(ds)
    print(
        f"PASS. train={len(idx['train_indices'])} val={len(idx['val_indices'])} "
        f"shap={len(idx['shap_indices'])}  positive rate={pos:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
