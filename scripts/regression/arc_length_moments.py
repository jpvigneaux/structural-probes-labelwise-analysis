#!/usr/bin/env python3
"""Moments of the linear distance distribution, per dependency relation.

Linear distance is the linear distance between a head and its dependent, in tokens,
following Gildea & Jaeger (2015). For each relation this reports the count, and
both the raw and the log-scale moments:

    n_instances  mean_length  stdev_length  median_length
    mean_log_length  stdev_log_length  skew_log_length

The paper's regression uses the log-scale ones. `mean_log_length` and
`stdev_log_length` are two of its three predictors, and `skew_log_length` is the
third rung of the moment ladder in Appendix "Moments of the length
distribution"; see compare_length_predictors.py for why the mean of the log
rather than the log of the mean, compare_dispersion_scale.py for why the
standard deviation rather than the variance, and add_length_spread.py for
whether the skewness earns its parameter (it does not).

Root arcs (head index 0) have no meaningful linear distance and are excluded
unless --include-root is passed, which assigns them the dependent's position.

The reader takes columns 1, 6 and 7 of a tab-separated file, so it accepts both
CoNLL-U and the CoNLL-X the Penn Treebank splits are in. The paper computes
these on PTB train.

Usage:
    python arc_length_moments.py <corpus.conllx> [--output <out.tsv>] [--include-root]
"""

import argparse
import math
import sys
from collections import defaultdict


def parse_conll(path: str):
    """Yield sentences as lists of token dicts with id, form, head, deprel.

    Columns 1/6/7 (form/head/deprel) are common to CoNLL-U and CoNLL-X, so this
    reads both. Multi-word and empty-node ids ('3-4', '3.1') are skipped.
    """
    sentence = []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if sentence:
                    yield sentence
                    sentence = []
                continue
            fields = line.split("\t")
            if len(fields) < 8:
                continue
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            sentence.append({
                "id":     int(token_id),
                "form":   fields[1],
                "head":   fields[6],
                "deprel": fields[7].lower(),
            })
    if sentence:
        yield sentence


def collect_lengths(corpus_path: str, include_root: bool = False):
    """
    Return {deprel: [length, length, ...]} where length = |head_pos - dep_pos|.
    Root arcs (head index 0) are excluded unless include_root is True,
    in which case they are assigned length = dep_pos (distance from position 0).
    """
    lengths: dict[str, list[int]] = defaultdict(list)

    for sentence in parse_conll(corpus_path):
        id2pos = {tok["id"]: tok["id"] for tok in sentence}  # 1-indexed positions

        for tok in sentence:
            deprel = tok["deprel"]
            head_str = tok["head"]

            if not head_str.lstrip("-").isdigit():
                continue
            head_id = int(head_str)

            if head_id == 0:
                if include_root:
                    length = tok["id"]  # distance from virtual root at position 0
                    lengths[deprel].append(length)
                continue

            if head_id not in id2pos:
                continue

            length = abs(id2pos[head_id] - tok["id"])
            lengths[deprel].append(length)

    return lengths


def mlog(xs):
    return sum(math.log(x) for x in xs) / len(xs)

def mean(xs):
    return sum(xs) / len(xs)

def stdev_log(xs):
    """SD of log linear distance.

    Complements `mean_log_length`: it measures how spread out a relation is on
    the scale on which UASL actually decays (the log-linear decay model of the
    linear distance analysis), whereas `stdev_length` measures spread of the raw
    lengths and is dominated by the long right tail.
    """
    return stdev([math.log(x) for x in xs])

def skew_log(xs):
    """Adjusted Fisher-Pearson skewness of log linear distance.

    Third standardised moment, continuing the sequence mean / sd / skew. The
    reason to look past the mean is exact rather than asymptotic: if UASL were
    an exactly linear function f of log length, a relation's mean UASL would be
    f(mean log length) identically and no other feature of its length
    distribution could carry information. Since the log-linear decay model
    holds only approximately, later moments are worth testing.

    Note that a Taylor expansion of E[f(X)] would suggest entering dispersion
    as a variance; `compare_dispersion_scale.py` shows the standard deviation
    to fit better on every model tested, so the expansion is not a reliable
    guide to the parametrisation and is not relied on here.
    """
    logs = [math.log(x) for x in xs]
    n = len(logs)
    if n < 3:
        return 0.0
    m = mean(logs)
    m2 = sum((x - m) ** 2 for x in logs) / n
    m3 = sum((x - m) ** 3 for x in logs) / n
    if m2 <= 0:
        return 0.0
    g1 = m3 / (m2 ** 1.5)
    return math.sqrt(n * (n - 1)) / (n - 2) * g1

def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

def median(xs):
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    parser = argparse.ArgumentParser(
        description="Compute mean dependency length by relation from a CoNLL-U corpus."
    )
    parser.add_argument("corpus", help="Path to the CoNLL-U file")
    parser.add_argument("--output", "-o", default=None,
                        help="Write TSV output to this file (default: stdout)")
    parser.add_argument("--include-root", action="store_true",
                        help="Include root arcs (head=0) in the counts")
    args = parser.parse_args()

    print(f"Reading corpus: {args.corpus}", file=sys.stderr)
    lengths = collect_lengths(args.corpus, args.include_root)
    print(f"Relations found: {len(lengths)}", file=sys.stderr)

    rows = []
    for deprel, lens in lengths.items():
        rows.append((
            deprel,
            len(lens),
            mean(lens),
            mlog(lens),
            stdev(lens),
            stdev_log(lens),
            skew_log(lens),
            median(lens),
        ))

    # Sort by mean dependency length descending
    rows.sort(key=lambda r: r[2], reverse=True)

    header = "\t".join([
        "deprel", "n_instances", "mean_length", "mean_log_length", "stdev_length",
        "stdev_log_length", "skew_log_length", "median_length"
    ])
    lines = [header] + [
        f"{deprel}\t{n}\t{mu:.4f}\t{mlu:.4f}\t{sd:.4f}\t{sdl:.4f}\t{sk:.4f}\t{med:.1f}"
        for deprel, n, mu, mlu, sd, sdl, sk, med in rows
    ]
    output_text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output_text)
        print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
