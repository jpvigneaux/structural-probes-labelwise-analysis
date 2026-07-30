#!/usr/bin/env python3
"""Should dispersion of log arc length enter as the variance or the SD?

The Taylor expansion that motivates a dispersion term,

    E[f(X)] = f(mu) + f''(mu) sigma^2 / 2 + f'''(mu) skew sigma^3 / 6 + ...,

contains sigma^2, not sigma. If the expansion is the reason for including the
term, the variance is the theoretically indicated regressor and the standard
deviation is a monotone reparametrisation of it that the expansion does not
license. This script asks whether the data prefer either.

The two models are non-nested but have the same number of predictors, are
fitted on the same relations with the same weights, and predict the same
response, so R^2, adjusted R^2, AIC and the log-likelihood are directly
comparable between them. A paired comparison of the per-relation weighted
squared residuals is also reported.

A third fit includes both sigma and sigma^2, which *does* nest each of the
other two, so the two nested F-tests say whether either scale carries
information the other lacks.

Usage:
    python compare_dispersion_scale.py -uuas dev.uuas_by_relation \\
        -sim results_sim_ptb.tsv -len dep-lengths-ptb.tsv [--label MODEL]
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

BASE = ['head_sim_entropy', 'mean_log_length']


def load(uuas_path, sim_path, len_path):
    uuas = pd.read_csv(uuas_path, sep='\t')
    if 'total' not in uuas.columns and {'correct', 'uuas'}.issubset(uuas.columns):
        uuas['total'] = uuas['correct'] / uuas['uuas']
    uuas = uuas.set_index('relation')
    sim = pd.read_csv(sim_path, sep='\t').set_index('deprel')
    length = pd.read_csv(len_path, sep='\t').set_index('deprel')

    common = uuas.index.intersection(sim.index).intersection(length.index)
    sd = length.loc[common, 'stdev_log_length']
    return pd.DataFrame({
        'uuas': uuas.loc[common, 'uuas'],
        'total': uuas.loc[common, 'total'],
        'head_sim_entropy': sim.loc[common, 'head_sim_entropy_bits'],
        'mean_log_length': length.loc[common, 'mean_log_length'],
        'sd_log_length': sd,
        'var_log_length': sd ** 2,
    }).dropna()


def fit(df, cols):
    X = sm.add_constant(df[cols].to_numpy())
    return sm.WLS(df['uuas'].to_numpy(), X, weights=df['total'].to_numpy()).fit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-uuas', required=True)
    ap.add_argument('-sim', required=True)
    ap.add_argument('-len', dest='length', required=True)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    df = load(args.uuas, args.sim, args.length)
    r = df['sd_log_length'].corr(df['var_log_length'])
    print(f'=== {args.label} ===  n = {len(df)}   corr(sd, var) = {r:+.4f}')

    m_sd = fit(df, BASE + ['sd_log_length'])
    m_var = fit(df, BASE + ['var_log_length'])
    m_both = fit(df, BASE + ['sd_log_length', 'var_log_length'])

    for name, m in (('sd ', m_sd), ('var', m_var)):
        print(f'  {name}  R^2 = {m.rsquared:.4f}  adjR^2 = {m.rsquared_adj:.4f}  '
              f'AIC = {m.aic:.2f}  logL = {m.llf:.3f}  '
              f'beta = {m.params[-1]:+.4f} (SE {m.bse[-1]:.4f}, p = {m.pvalues[-1]:.4f})')

    d_r2 = m_sd.rsquared - m_var.rsquared
    d_aic = m_sd.aic - m_var.aic
    winner = 'sd' if d_r2 > 0 else 'var'
    print(f'\n  delta R^2 (sd - var) = {d_r2:+.4f}   '
          f'delta AIC (sd - var) = {d_aic:+.2f}   -> {winner} fits better')

    w = df['total'].to_numpy()
    diff = w * (m_var.resid ** 2) - w * (m_sd.resid ** 2)   # >0 where sd wins
    print(f'  sd has the smaller weighted squared residual on '
          f'{int((diff > 0).sum())}/{len(df)} relations')

    print('\n  both sigma and sigma^2 in the same model:')
    for i, nm in enumerate(['const'] + BASE + ['sd_log_length', 'var_log_length']):
        print(f'    {nm:<18s} beta = {m_both.params[i]:+.4f}  '
              f'SE = {m_both.bse[i]:.4f}  p = {m_both.pvalues[i]:.4f}')
    for added, reduced in (('sd_log_length', m_var), ('var_log_length', m_sd)):
        f_stat, p_val, _ = m_both.compare_f_test(reduced)
        print(f'    adding {added:<15s} to the other: F = {f_stat:.3f}, p = {p_val:.4f}')


if __name__ == '__main__':
    main()
