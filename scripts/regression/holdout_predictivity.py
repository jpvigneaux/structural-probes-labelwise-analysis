#!/usr/bin/env python3
"""Does the regression predict UASL, or only describe the split it was fitted on?

The regression of the paper is fitted on 42 numbers with three predictors, which
invites the question of whether its R^2 is a measure of anything beyond its own
flexibility. There are two separate ways it could fail, and they need separate
tests, because a figure that answers one says nothing about the other.

  Held-out edges.  Fit on dev UASL, then predict the UASL of the same relations
    as re-measured on the test split. This asks whether the fit is tracking the
    relations or the particular sample of dev edges their UASL was computed
    from. The predictors come from ptb3-wsj-train.conllx, so train, dev and test
    are disjoint and nothing about the test edges enters the fit.

  Held-out relations.  Leave one relation out, refit on the remaining 41, and
    predict the omitted one. This is the test that speaks to the ratio of
    parameters to observations, and it is the harder of the two.

Both are reported as weighted R^2 against the weighted mean, 1 - SSres/SStot, so
a model no better than the intercept scores 0 and a worse one scores below 0 --
unlike a squared correlation, which cannot.

The two numbers should be read together with r(dev, test): dev and test UASL are
two measurements of one quantity, so a high held-out-edge R^2 is partly assured
in advance, and the smaller that correlation the more the first test is telling
you.

Usage:
    python holdout_predictivity.py --spec "BERT-base:RESULTS:16" \\
        --sim sim_static_fasttext.tsv --len dep-lengths-ptb.tsv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _regdata import load_regression_frame          # noqa: E402

KEYS = ['head_sim_entropy', 'mean_log_length', 'sd_log_length']


def wls_beta(y, X, w):
    A = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return beta


def predict(beta, X):
    return np.column_stack([np.ones(len(X)), X]) @ beta


def weighted_r2(y, yhat, w):
    """1 - SSres/SStot, weighted. Unbounded below, unlike a squared correlation."""
    ybar = np.sum(w * y) / np.sum(w)
    return 1 - np.sum(w * (y - yhat) ** 2) / np.sum(w * (y - ybar) ** 2)


def heldout_frame(results, ck, sim_path, len_path):
    """The regression's own frame, with the test-split UASL joined on.

    The development side comes from load_regression_frame, so this is the same
    join the regression itself uses and cannot drift from it. Relations with no
    test-set edges are dropped and returned separately, since their absence is
    a fact about the corpus that the caption has to state.
    """
    layer = Path(results) / f'layer-{int(ck):02d}'
    df = load_regression_frame(layer / 'dev.uuas_by_relation', sim_path, len_path,
                               columns=KEYS + ['uuas', 'total'])
    t = pd.read_csv(layer / 'test.uuas_by_relation', sep='\t').set_index('relation')
    if 'total' not in t.columns:
        t['total'] = t['correct'] / t['uuas']
    dropped = sorted(set(df.index) - set(t.index))
    df = df.assign(test_uuas=t['uuas'].reindex(df.index),
                   test_total=t['total'].reindex(df.index))
    return df.dropna(subset=['test_uuas', 'test_total']), dropped


def heldout_stats(df):
    """Fit on dev, score on held-out edges and on held-out relations."""
    X = df[KEYS].to_numpy(dtype=float)
    y_dev, w_dev = df['uuas'].to_numpy(float), df['total'].to_numpy(float)
    y_test, w_test = df['test_uuas'].to_numpy(float), df['test_total'].to_numpy(float)

    beta = wls_beta(y_dev, X, w_dev)

    # Leave one relation out, refitting each time; the omitted relation is then
    # predicted by a model that never saw it.
    loo = np.empty(len(df))
    for i in range(len(df)):
        m = np.ones(len(df), bool)
        m[i] = False
        loo[i] = predict(wls_beta(y_dev[m], X[m], w_dev[m]), X[i:i + 1])[0]

    return {
        'n': len(df),
        'r2_dev': weighted_r2(y_dev, predict(beta, X), w_dev),
        'r2_test': weighted_r2(y_test, predict(beta, X), w_test),
        'q2_loo': weighted_r2(y_dev, loo, w_dev),
        'r_dev_test': float(np.corrcoef(y_dev, y_test)[0, 1]),
        'beta_entropy_dev': float(beta[1]),
        'beta_entropy_test': float(wls_beta(y_test, X, w_test)[1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', action='append', required=True,
                    help='LABEL:RESULTS_DIR:CHECKPOINT (repeatable)')
    ap.add_argument('--sim', required=True)
    ap.add_argument('--len', dest='length', required=True)
    args = ap.parse_args()

    print(f'{"model":17s} {"n":>3s} {"R2 dev":>8s} {"R2 test":>8s} {"Q2 LOO":>8s} '
          f'{"r(dev,test)":>12s} {"b_ent dev":>10s} {"b_ent test":>11s}')
    for spec in args.spec:
        label, results, ck = spec.rsplit(':', 2)
        df, dropped = heldout_frame(results, ck, args.sim, args.length)
        s = heldout_stats(df)
        print(f'{label:17s} {s["n"]:3d} {s["r2_dev"]:8.3f} {s["r2_test"]:8.3f} '
              f'{s["q2_loo"]:8.3f} {s["r_dev_test"]:12.4f} '
              f'{s["beta_entropy_dev"]:10.4f} {s["beta_entropy_test"]:11.4f}')
        if dropped:
            print(f'{"":17s}     (absent from the test split, dropped: '
                  f'{", ".join(dropped)})')


if __name__ == '__main__':
    main()
