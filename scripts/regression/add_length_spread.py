#!/usr/bin/env python3
"""Do higher moments of the arc-length distribution add explanatory power?

The published model uses only the location of a relation's log-length
distribution (`mean_log_length`). Its dispersion and asymmetry are natural
further candidates, and there is an exact reason to test them. Let f(x) be the
accuracy the probe attains on an edge of log length x, and X the log length of
a random edge of a given relation, so the relation's ULAS is E[f(X)]. If f were
exactly linear -- the log-linear decay model that the arc-length analysis shows
to be imperfect -- then E[f(X)] = f(E[X]) identically and the mean would
exhaust what the length distribution can contribute. It is not, so later
moments may carry information. This script fits the corresponding nested
ladder:

    M1  uuas ~ head_sim_entropy + mean_log_length                    (published)
    M2  uuas ~ ... + sd_log_length
    M3  uuas ~ ... + sd_log_length + skew_log_length

Each model nests the previous one, so each step is tested with an F-test on the
added term, reported with adjusted R^2 and AIC. Raw R^2 cannot decrease when a
predictor is added and is not evidence on its own. With only ~42 relations,
adjusted R^2 and AIC are the numbers to read.

Usage:
    python add_length_spread.py -uuas dev.uuas_by_relation \\
        -sim results_sim_ptb.tsv -len dep-lengths-ptb.tsv [--label MODEL]
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

LADDER = [
    ('M1 (published)', ['head_sim_entropy', 'mean_log_length']),
    ('M2 (+ sd)      ', ['head_sim_entropy', 'mean_log_length', 'sd_log_length']),
    ('M3 (+ sd,skew) ', ['head_sim_entropy', 'mean_log_length', 'sd_log_length',
                         'skew_log_length']),
]


def load(uuas_path, sim_path, len_path):
    uuas = pd.read_csv(uuas_path, sep='\t')
    if 'total' not in uuas.columns and {'correct', 'uuas'}.issubset(uuas.columns):
        uuas['total'] = uuas['correct'] / uuas['uuas']
    uuas = uuas.set_index('relation')
    sim = pd.read_csv(sim_path, sep='\t').set_index('deprel')
    length = pd.read_csv(len_path, sep='\t').set_index('deprel')
    for col in ('stdev_log_length', 'skew_log_length'):
        if col not in length.columns:
            raise SystemExit(f'length file lacks {col}; re-run ud_dep_length.py')

    common = uuas.index.intersection(sim.index).intersection(length.index)
    return pd.DataFrame({
        'uuas': uuas.loc[common, 'uuas'],
        'total': uuas.loc[common, 'total'],
        'head_sim_entropy': sim.loc[common, 'head_sim_entropy_bits'],
        'mean_log_length': length.loc[common, 'mean_log_length'],
        'sd_log_length': length.loc[common, 'stdev_log_length'],
        'skew_log_length': length.loc[common, 'skew_log_length'],
    }).dropna()


def fit(df, cols):
    X = sm.add_constant(df[cols].to_numpy())
    return sm.WLS(df['uuas'].to_numpy(), X, weights=df['total'].to_numpy()).fit()


def vif(df, cols):
    out = {}
    for c in cols:
        others = [x for x in cols if x != c]
        r2 = sm.OLS(df[c].to_numpy(),
                    sm.add_constant(df[others].to_numpy())).fit().rsquared
        out[c] = 1.0 / max(1e-12, 1.0 - r2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-uuas', required=True)
    ap.add_argument('-sim', required=True)
    ap.add_argument('-len', dest='length', required=True)
    ap.add_argument('--label', default='')
    ap.add_argument('--full', action='store_true',
                    help='print the full coefficient table for M3')
    args = ap.parse_args()

    df = load(args.uuas, args.sim, args.length)
    print(f'=== {args.label} ===  n = {len(df)}')

    fits = []
    for name, cols in LADDER:
        m = fit(df, cols)
        fits.append((name, cols, m))
        print(f'  {name}  R^2 = {m.rsquared:.4f}  adjR^2 = {m.rsquared_adj:.4f}  '
              f'AIC = {m.aic:.2f}')

    print()
    for (n_prev, _, m_prev), (n_next, cols_next, m_next) in zip(fits, fits[1:]):
        f_stat, p_val, _ = m_next.compare_f_test(m_prev)
        added = cols_next[-1]
        beta, se, p_coef = (m_next.params[-1], m_next.bse[-1], m_next.pvalues[-1])
        print(f'  adding {added:<16s} beta = {beta:+.4f} (SE {se:.4f}, p = {p_coef:.3e})')
        print(f'    F = {f_stat:.3f}, p = {p_val:.4f}   '
              f'delta adjR^2 = {m_next.rsquared_adj - m_prev.rsquared_adj:+.4f}   '
              f'delta AIC = {m_next.aic - m_prev.aic:+.2f}   '
              f'-> {"KEEP" if p_val < 0.05 else "drop"}')

    m3 = fits[-1][2]
    cols3 = fits[-1][1]
    if args.full:
        print('\n  M3 coefficients:')
        for i, nm in enumerate(['const'] + cols3):
            print(f'    {nm:<18s} beta = {m3.params[i]:+.4f}  SE = {m3.bse[i]:.4f}  '
                  f'p = {m3.pvalues[i]:.3e}')
    print('  VIF(M3): ' + '  '.join(f'{k}={v:.2f}' for k, v in vif(df, cols3).items()))
    print(f'  corr(sd, skew) = '
          f'{df["sd_log_length"].corr(df["skew_log_length"]):+.3f}')


if __name__ == '__main__':
    main()
