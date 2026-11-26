#!/usr/bin/env python3
"""
Step 1 — Generate 20 stratified random splits of the PTB.

Reads the three original CoNLL-X files (train/dev/test), pools every sentence,
then re-partitions into 20 new train/dev/test splits that:
  • preserve the original proportions (# sentences),
  • are stratified by a (dominant_dep_relation, length_bucket) compound label
    so that every syntactic relation type is approximately equally represented
    across all splits.

Outputs
-------
  data/split_NN/{train,dev,test}.conllx   — 60 CoNLL-X files
  split_metadata.json                      — provenance: which original split
                                             and sentence index each new
                                             sentence came from; consumed by
                                             make_hdf5_splits.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# ── Source data (original PTB split) ─────────────────────────────────────────
# DATA_DIR as set in paths.sh (the directory holding the PTB .conllx files).
_DATA = os.environ.get("DATA_DIR")
if not _DATA:
    sys.exit("set DATA_DIR (source paths.sh) to the directory holding ptb3-wsj-*.conllx")
ORIG = {split: os.path.join(_DATA, f"ptb3-wsj-{split}.conllx")
        for split in ("train", "dev", "test")}

N_SPLITS = 20

# Base seed for split generation (reproducible; distinct from probe training seeds)
SPLIT_GEN_SEED = 2024


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_conllx(path):
    """Return list of sentences; each sentence is a list of raw tab-delimited lines."""
    sentences, current = [], []
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(line)
    if current:
        sentences.append(current)
    return sentences


def strat_label(sentence):
    """
    Return a compound stratification label for one sentence.

    Label = "<dominant_dep_relation>|<length_bucket>"

    Length buckets: S (≤10 tokens), M (11–30), L (>30).
    Using a compound label ensures that both the syntactic-relation distribution
    AND the sentence-length distribution are approximately preserved across splits.
    """
    relations = []
    for line in sentence:
        fields = line.split("\t")
        if len(fields) >= 8:
            relations.append(fields[7])
    n = len(sentence)
    dominant = Counter(relations).most_common(1)[0][0] if relations else "UNK"
    bucket = "S" if n <= 10 else ("M" if n <= 30 else "L")
    return f"{dominant}|{bucket}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    here = Path(__file__).parent

    # Load all sentences, remembering which original file each came from
    all_sentences: list[list[str]] = []
    provenance: list[tuple[str, int]] = []  # (original_split_name, original_sent_idx)

    for split_name in ("train", "dev", "test"):
        sents = read_conllx(ORIG[split_name])
        print(f"  {split_name}: {len(sents)} sentences")
        for i, s in enumerate(sents):
            all_sentences.append(s)
            provenance.append((split_name, i))

    n_total = len(all_sentences)
    print(f"Total sentences pooled: {n_total}")

    # Compute target sizes from original proportions (round to nearest int)
    orig_counts = {k: len(read_conllx(v)) for k, v in ORIG.items()}
    n_dev_target  = orig_counts["dev"]
    n_test_target = orig_counts["test"]
    dev_frac  = n_dev_target  / n_total
    test_frac = n_test_target / n_total
    print(f"Target sizes  train≈{n_total - n_dev_target - n_test_target}  "
          f"dev≈{n_dev_target}  test≈{n_test_target}")

    # Build stratification labels
    raw_labels = [strat_label(s) for s in all_sentences]

    # Collapse compound labels that appear fewer than MIN_PER_LABEL times into
    # their dominant-relation-only label (keeps StratifiedShuffleSplit happy).
    MIN_PER_LABEL = 40
    label_counts = Counter(raw_labels)
    labels = np.array([
        lbl if label_counts[lbl] >= MIN_PER_LABEL else lbl.split("|")[0]
        for lbl in raw_labels
    ])
    # Second pass: collapse again if still too rare after stripping bucket
    label_counts2 = Counter(labels)
    labels = np.array([
        lbl if label_counts2[lbl] >= MIN_PER_LABEL else "OTHER"
        for lbl in labels
    ])
    print(f"Stratification labels: {len(set(labels))} distinct classes")

    idx_all = np.arange(n_total)
    metadata: dict = {}

    for split_idx in range(N_SPLITS):
        seed_test = SPLIT_GEN_SEED + split_idx * 1000
        seed_dev  = SPLIT_GEN_SEED + split_idx * 1000 + 1

        # Carve out test set
        sss_test = StratifiedShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=seed_test
        )
        traindev_idx, test_idx = next(sss_test.split(idx_all, labels))

        # Carve dev from the remaining traindev pool
        dev_frac_of_traindev = dev_frac / (1.0 - test_frac)
        sss_dev = StratifiedShuffleSplit(
            n_splits=1, test_size=dev_frac_of_traindev, random_state=seed_dev
        )
        labels_traindev = labels[traindev_idx]
        rel_train_idx, rel_dev_idx = next(
            sss_dev.split(traindev_idx, labels_traindev)
        )
        train_idx = traindev_idx[rel_train_idx]
        dev_idx   = traindev_idx[rel_dev_idx]

        split_key = f"split_{split_idx:02d}"
        print(f"{split_key}: train={len(train_idx)}  dev={len(dev_idx)}  test={len(test_idx)}")

        # Write CoNLL-X files
        data_dir = here / "data" / split_key
        data_dir.mkdir(parents=True, exist_ok=True)

        split_meta: dict = {}
        for subset_name, indices in [
            ("train", train_idx),
            ("dev",   dev_idx),
            ("test",  test_idx),
        ]:
            out_path = data_dir / f"{subset_name}.conllx"
            with open(out_path, "w") as fh:
                for orig_idx in indices:
                    for line in all_sentences[orig_idx]:
                        fh.write(line + "\n")
                    fh.write("\n")
            # Store provenance as list of [split_name, sent_idx] (JSON-serialisable)
            split_meta[subset_name] = [list(provenance[i]) for i in indices]

        metadata[split_key] = split_meta

    meta_path = here / "split_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh)
    print(f"\nSaved provenance to {meta_path}")


if __name__ == "__main__":
    main()
