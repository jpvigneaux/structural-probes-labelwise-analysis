#!/usr/bin/env python3
"""Compare two ways of summarising a relation's linear distance in the WLS regression.

    mean_log_length  = mean_i log(n_i)      -- log of the GEOMETRIC mean
    log_mean_length  = log(mean_i n_i)      -- log of the ARITHMETIC mean

By Jensen's inequality log(mean) >= mean(log), with the gap growing in the
variance of the length distribution. The two therefore differ most for relations
whose linear distances are widely spread, so the question is empirical: which summary
better predicts UASL?

The two models are non-nested but have identical predictor counts and are fitted
on identical data with identical weights, so R^2, AIC and the log-likelihood are
directly comparable between them. A paired comparison of the per-relation
residuals is also reported, since the models are fitted on the same 42 points.

Usage:
    python compare_length_predictors.py -uuas dev.uuas_by_relation \\
        -sim results_sim_ptb.tsv -len dep-lengths-ptb.tsv [--label MODEL]
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm


def load(uuas_path, sim_path, len_path):
    uuas = pd.read_csv(uuas_path, sep='\t')
    if 'total' not in uuas.columns and {'correct', 'uuas'}.issubset(uuas.columns):
        uuas['total'] = uuas['correct'] / uuas['uuas']
    uuas = uuas.set_index('relation')
    sim = pd.read_csv(sim_path, sep='\t').set_index('deprel')
    length = pd.read_csv(len_path, sep='\t').set_index('deprel')

    common = uuas.index.intersection(sim.index).intersection(length.index)
    df = pd.DataFrame({
        'uuas': uuas.loc[common, 'uuas'],
        'total': uuas.loc[common, 'total'],
        'head_sim_entropy': sim.loc[common, 'head_sim_entropy_bits'],
        'mean_log_length': length.loc[common, 'mean_log_length'],
        'log_mean_length': np.log(length.loc[common, 'mean_length']),
    }).dropna()
    return df


def fit(df, length_col):
    X = sm.add_constant(df[['head_sim_entropy', length_col]].to_numpy())
    return sm.WLS(df['uuas'].to_numpy(), X, weights=df['total'].to_numpy()).fit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-uuas', required=True)
    ap.add_argument('-sim', required=True)
    ap.add_argument('-len', dest='length', required=True)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    df = load(args.uuas, args.sim, args.length)
    r = np.corrcoef(df['mean_log_length'], df['log_mean_length'])[0, 1]

    header = f'=== {args.label} ===' if args.label else '==='
    print(f'{header}  n = {len(df)} relations; '
          f'predictor correlation r = {r:.4f}')

    results = {}
    for col in ('mean_log_length', 'log_mean_length'):
        m = fit(df, col)
        results[col] = m
        print(f'\n  {col}')
        print(f'    R^2 = {m.rsquared:.4f}   adj R^2 = {m.rsquared_adj:.4f}   '
              f'AIC = {m.aic:.2f}   logL = {m.llf:.3f}')
        print(f'    head_sim_entropy  beta = {m.params[1]:+.4f}  '
              f'SE = {m.bse[1]:.4f}  p = {m.pvalues[1]:.2e}')
        print(f'    {col:<17s} beta = {m.params[2]:+.4f}  '
              f'SE = {m.bse[2]:.4f}  p = {m.pvalues[2]:.2e}')

    a, b = results['mean_log_length'], results['log_mean_length']
    d_r2 = b.rsquared - a.rsquared
    better = 'log_mean_length' if d_r2 > 0 else 'mean_log_length'
    print(f'\n  delta R^2 (log_mean - mean_log) = {d_r2:+.4f}   '
          f'delta AIC = {b.aic - a.aic:+.2f}   -> {better} fits better')

    # Paired comparison of weighted squared residuals on the same relations.
    w = df['total'].to_numpy()
    ra, rb = a.resid, b.resid
    diff = w * (ra ** 2) - w * (rb ** 2)          # >0 where log_mean does better
    n_better = int((diff > 0).sum())
    print(f'  log_mean_length has the smaller weighted squared residual on '
          f'{n_better}/{len(df)} relations')

    # Which relations separate the two models most.
    df2 = df.assign(gap=df['log_mean_length'] - df['mean_log_length'],
                    resid_diff=diff)
    top = df2.reindex(df2['resid_diff'].abs().sort_values(ascending=False).index).head(5)
    print('\n  relations where the choice matters most:')
    print('    ' + top[['uuas', 'total', 'mean_log_length',
                        'log_mean_length', 'gap']].to_string().replace('\n', '\n    '))


if __name__ == '__main__':
    main()
