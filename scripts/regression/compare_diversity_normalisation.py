#!/usr/bin/env python3
"""Should the diversity predictor's log-argument be normalised?

The paper defines the head's similarity-corrected entropy as

    q(x)     = sum_y Z_xy p(y)                       [Leinster-Cobbold ordinariness]
    H_sim(r) = -sum_x p(x) log2 q(x)

and that is what contextual_sim_entropy.py computes. A natural variant divides
the log-argument by the relation's mean within-relation similarity,

    zpp(r)   = sum_x sum_y Z_xy p(x) p(y) = sum_x p(x) q(x),
    H_norm(r) = -sum_x p(x) log2 (q(x) / zpp(r)) = H_sim(r) + log2 zpp(r),

which makes the quantity invariant to a relation's overall similarity level and
measures only how unevenly that similarity is distributed. The two differ by an
additive term that is NOT constant across relations -- relations whose heads are
mutually similar have larger zpp -- so which one predicts UASL better is an
empirical question, not a matter of convention.

Three fits are compared, all with the same two companion predictors, the same 42
relations and the same weights, so R^2, adjusted R^2 and AIC are directly
comparable:

    published   H_sim              (the paper's definition)
    normalised  H_sim + log2 zpp
    Shannon     plain entropy, the Z = I limit, as a floor

A fourth fit enters H_sim and log2 zpp together, which nests the first two, so a
nested F-test says whether the normaliser carries information the published
predictor lacks.

Usage:
    python compare_diversity_normalisation.py -uuas dev.uuas_by_relation \\
        -sim sim_static_fasttext.tsv -len dep-lengths-ptb.tsv --label "BERT-base"
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm

LENGTH = ['mean_log_length', 'sd_log_length']


def load(uuas_path, sim_path, len_path):
    uuas = pd.read_csv(uuas_path, sep='\t')
    if 'total' not in uuas.columns and {'correct', 'uuas'}.issubset(uuas.columns):
        uuas['total'] = uuas['correct'] / uuas['uuas']
    uuas = uuas.set_index('relation')
    sim = pd.read_csv(sim_path, sep='\t').set_index('deprel')
    length = pd.read_csv(len_path, sep='\t').set_index('deprel')

    if 'head_zpp' not in sim.columns:
        raise SystemExit(
            f'{sim_path} lacks head_zpp. Re-run contextual_sim_entropy.py; it now '
            f'emits the normaliser alongside the entropy.')

    common = uuas.index.intersection(sim.index).intersection(length.index)
    published = sim.loc[common, 'head_sim_entropy_bits']
    zpp = sim.loc[common, 'head_zpp'].clip(lower=1e-12)
    return pd.DataFrame({
        'uuas': uuas.loc[common, 'uuas'],
        'total': uuas.loc[common, 'total'],
        'published': published,
        'normalised': published + np.log2(zpp),
        'shannon': sim.loc[common, 'head_entropy_bits'],
        'log2_zpp': np.log2(zpp),
        'mean_log_length': length.loc[common, 'mean_log_length'],
        'sd_log_length': length.loc[common, 'stdev_log_length'],
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
    r = df['published'].corr(df['normalised'])
    print(f'=== {args.label} ===  n = {len(df)}   '
          f'corr(published, normalised) = {r:+.4f}')
    print(f'    log2 zpp ranges over [{df["log2_zpp"].min():+.3f}, '
          f'{df["log2_zpp"].max():+.3f}] bits, sd {df["log2_zpp"].std():.3f} '
          f'-- the two definitions are not a constant apart')

    fits = {}
    for name in ('published', 'normalised', 'shannon'):
        m = fit(df, [name] + LENGTH)
        fits[name] = m
        print(f'  {name:<11s} R^2 = {m.rsquared:.4f}  adjR^2 = {m.rsquared_adj:.4f}  '
              f'AIC = {m.aic:.2f}   beta = {m.params[1]:+.4f}  p = {m.pvalues[1]:.4f}')

    best = max(fits, key=lambda k: fits[k].rsquared)
    d = fits['normalised'].rsquared - fits['published'].rsquared
    print(f'\n  delta R^2 (normalised - published) = {d:+.4f}   '
          f'delta AIC = {fits["normalised"].aic - fits["published"].aic:+.2f}'
          f'   -> {best} fits best')

    # Both terms at once: nests each single-term fit, so the F-tests say whether
    # either carries information the other lacks.
    both = fit(df, ['published', 'log2_zpp'] + LENGTH)
    print(f'\n  both terms:  R^2 = {both.rsquared:.4f}  '
          f'adjR^2 = {both.rsquared_adj:.4f}')
    for i, nm in enumerate(['published', 'log2_zpp'], start=1):
        print(f'    {nm:<11s} beta = {both.params[i]:+.4f}  '
              f'SE = {both.bse[i]:.4f}  p = {both.pvalues[i]:.4f}')
    f_stat, p_val, _ = both.compare_f_test(fits['published'])
    print(f'    adding log2 zpp to the published model: '
          f'F = {f_stat:.3f}, p = {p_val:.4f}')


if __name__ == '__main__':
    main()
