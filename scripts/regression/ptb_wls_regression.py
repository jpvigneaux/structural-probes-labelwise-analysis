#!/usr/bin/env python3
"""
Weighted least squares regression of per-relation UUAS on head-similarity
entropy, mean log dependency length and the standard deviation of log
dependency length, using fully native PTB-scheme data throughout
(dev.uuas_by_relation, results_sim_ptb.tsv, dep-lengths-ptb.tsv).

These three predictors are the published model. `--no-sd` drops the dispersion
term, recovering the two-predictor model of the earlier version of the analysis;
see `add_length_spread.py` for the nested test that motivates keeping it and
`compare_dispersion_scale.py` for why it enters as a standard deviation.

No label mapping/translation is needed or performed: all three files must
already share the same dependency-relation labels, so relations are joined
directly on that column. This avoids the data loss (and directional
ambiguity) that comes from mapping between different relation schemes
(e.g. PTB vs UD) -- if the files you pass in use different label schemes,
the join will simply come up mostly (or entirely) empty rather than
silently mismatching relations.

Regression is weighted by `total` (the number of dev-set instances each
relation's UUAS was computed from), since UUAS is a proportion and its
sampling noise depends on that denominator -- relations evaluated on only
a handful of tokens (e.g. csubjpass, n=1) get appropriately down-weighted
relative to high-frequency relations (e.g. prep, n=3783).

Usage:
    python3 ptb_wls_regression.py -uuas dev.uuas_by_relation \\
                                   -sim results_sim_ptb.tsv \\
                                   -len dep-lengths-ptb.tsv
"""

import argparse
import os
import sys

import pandas as pd
import statsmodels.api as sm


def parse_args():
    parser = argparse.ArgumentParser(
        description="WLS regression of UUAS on head_sim_entropy and mean_log_length."
    )
    parser.add_argument(
        "-uuas", type=str, required=True,
        help="Path to the UUAS-by-relation file (TSV). "
             "Expected columns: relation, uuas, total.",
    )
    parser.add_argument(
        "-sim", type=str, required=True,
        help="Path to the similarity/entropy file (TSV). "
             "Expected columns: deprel, head_sim_entropy_bits.",
    )
    parser.add_argument(
        "-len", type=str, required=True,
        help="Path to the dependency-length file (TSV). "
             "Expected columns: deprel, mean_log_length, stdev_log_length.",
    )
    parser.add_argument(
        "--no-sd", action="store_true",
        help="Omit sd(log length), recovering the two-predictor model.",
    )
    parser.add_argument(
        "--label", type=str, default="",
        help="Model name, printed with the summary.",
    )
    return parser.parse_args()


def load_uuas(path):
    df = pd.read_csv(path, sep="\t")
    if "total" not in df.columns and {"correct", "uuas"}.issubset(df.columns):
        df["total"] = df["correct"] / df["uuas"]
    return df.set_index("relation")


def main():
    args = parse_args()

    for path, flag in [(args.uuas, "-uuas"), (args.sim, "-sim"), (args.len, "-len")]:
        if not os.path.exists(path):
            print(f"Error: file for {flag} not found: {path}", file=sys.stderr)
            sys.exit(1)

    uuas = load_uuas(args.uuas)
    sim = pd.read_csv(args.sim, sep="\t").set_index("deprel")
    length = pd.read_csv(args.len, sep="\t").set_index("deprel")

    # Direct join on shared relation labels -- no relabeling performed.
    common = uuas.index.intersection(sim.index).intersection(length.index)
    if len(common) == 0:
        print(
            "Error: no relations in common across the three files. "
            "Do they use the same dependency-relation label scheme?",
            file=sys.stderr,
        )
        sys.exit(1)

    df = pd.DataFrame({
        "uuas": uuas.loc[common, "uuas"],
        "total": uuas.loc[common, "total"],
        "head_sim_entropy": sim.loc[common, "head_sim_entropy_bits"],
        "mean_log_length": length.loc[common, "mean_log_length"],
    })
    predictors = ["head_sim_entropy", "mean_log_length"]
    if not args.no_sd:
        if "stdev_log_length" not in length.columns:
            print("Error: -len file lacks stdev_log_length; re-run "
                  "ud_dep_length.py, or pass --no-sd.", file=sys.stderr)
            sys.exit(1)
        df["sd_log_length"] = length.loc[common, "stdev_log_length"]
        predictors.append("sd_log_length")
    df = df.dropna()

    if args.label:
        print(f"=== {args.label} ===")
    print(f"Matched relations: {len(df)}")
    print(df.sort_values("total", ascending=False).to_string())

    X = sm.add_constant(df[predictors].values)
    y = df["uuas"].values
    weights = df["total"].values

    model = sm.WLS(y, X, weights=weights).fit()

    print("\n=== WLS Regression Summary ===")
    print(model.summary(xname=["const"] + predictors, yname="uuas"))

    return df, model


if __name__ == "__main__":
    main()
