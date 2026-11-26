#!/usr/bin/env python3
"""Robustness of the three-predictor regression, for the ARR 2026-08 author response.

Every block refits the published model (Table `tab:regression_results`) under one
change and prints the head-entropy coefficient first, since that is the effect the
reviewers question. Blocks:

  weights     how relations are weighted (r12d W2): WLS by n (published), OLS,
              sqrt(n), binomial inverse variance n / (p(1-p)), a binomial GLM on
              (correct, total) with logit link, dropping the three heaviest
              relations, dropping relations with n < 20, HC3 standard errors,
              Cook's distance and the leave-one-relation-out range of beta_ent.
  dependent   adds dependent similarity-aware entropy (r12d W3): VIFs, the
              head/dependent correlation, and the two coefficients side by side.
  confounds   adds, one at a time and all together (U2Ge Q3): log frequency of
              the relation in PTB train, plain Shannon head entropy, log number of
              distinct heads, Shannon entropy of the head's POS, share of verbal
              heads. VIFs for the full model.
  alt-sim     swaps the similarity file (U2Ge Q4, r12d Q2): any number of
              --alt-sim LABEL=PATH, e.g. fastText at other PCA dimensions, GloVe.
  replicates  refits on many probing runs of the same model (all reviewers):
              --replicates GLOB of dev.uuas_by_relation files; prints the spread
              of each coefficient and its sign/significance counts.

Usage:
    python rebuttal_robustness.py --uuas DEV_UASL --sim SIM --len LEN \
        --conllx ptb3-wsj-train.conllx [--alt-sim L=P ...] [--replicates GLOB]
"""

import argparse
import glob
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _regdata import load_regression_frame          # noqa: E402

BASE = ['head_sim_entropy', 'mean_log_length', 'sd_log_length']
pd.set_option('display.width', 160)


def fit(df, preds, w=None, cov='nonrobust'):
    X = sm.add_constant(df[preds].values)
    m = (sm.WLS(df['uuas'].values, X, weights=w) if w is not None
         else sm.OLS(df['uuas'].values, X)).fit(cov_type=cov)
    return m


def row(label, m, preds, n):
    out = {'model': label, 'n': n, 'R2': round(m.rsquared, 3)}
    for i, p in enumerate(preds, start=1):
        out[p] = f'{m.params[i]:+.3f} (p={m.pvalues[i]:.3g})'
    return out


def vif_table(df, preds):
    X = sm.add_constant(df[preds].values)
    return {p: round(variance_inflation_factor(X, i), 2) for i, p in enumerate(preds, start=1)}


def corpus_controls(conllx):
    """Per relation, from PTB train: head-POS entropy and share of verbal heads."""
    pos_counts = defaultdict(Counter)
    sent = []

    def flush(s):
        for tok in s:
            head = int(tok[6])
            if head == 0:
                continue
            pos_counts[tok[7]][s[head - 1][4]] += 1

    for line in open(conllx, encoding='utf-8'):
        line = line.rstrip('\n')
        if not line:
            if sent:
                flush(sent)
            sent = []
            continue
        f = line.split('\t')
        if '-' in f[0] or '.' in f[0]:
            continue
        sent.append(f)
    if sent:
        flush(sent)
    rows = {}
    for rel, c in pos_counts.items():
        tot = sum(c.values())
        p = np.array(list(c.values())) / tot
        rows[rel] = {'head_pos_entropy': float(-(p * np.log2(p)).sum()),
                     'verb_head_share': sum(v for k, v in c.items() if k.startswith('VB')) / tot}
    return pd.DataFrame.from_dict(rows, orient='index')


def with_extras(df, sim_path, len_path, conllx):
    sim = pd.read_csv(sim_path, sep='\t').set_index('deprel')
    length = pd.read_csv(len_path, sep='\t').set_index('deprel')
    ctl = corpus_controls(conllx)
    df = df.copy()
    df['dep_sim_entropy'] = sim.loc[df.index, 'dep_sim_entropy_bits']
    df['head_shannon_entropy'] = sim.loc[df.index, 'head_entropy_bits']
    df['log_n_head_types'] = np.log(sim.loc[df.index, 'n_distinct_heads'])
    df['log_train_freq'] = np.log(length.loc[df.index, 'n_instances'])
    df['head_pos_entropy'] = ctl.loc[df.index, 'head_pos_entropy']
    df['verb_head_share'] = ctl.loc[df.index, 'verb_head_share']
    return df


def block_weights(df):
    print('\n### weights (r12d W2)')
    w = df['total'].values
    rows = [row('WLS, w = n (published)', fit(df, BASE, w), BASE, len(df)),
            row('OLS, unweighted', fit(df, BASE), BASE, len(df)),
            row('WLS, w = sqrt(n)', fit(df, BASE, np.sqrt(w)), BASE, len(df))]
    p = (df['uuas'] * df['total'] + 0.5) / (df['total'] + 1)
    rows.append(row('WLS, w = n / p(1-p)', fit(df, BASE, w / (p * (1 - p))), BASE, len(df)))
    rows.append(row('WLS, HC3 robust SE', fit(df, BASE, w, cov='HC3'), BASE, len(df)))
    heavy = df['total'].nlargest(3).index
    d2 = df.drop(heavy)
    rows.append(row(f'WLS, without {"/".join(heavy)}', fit(d2, BASE, d2['total'].values), BASE, len(d2)))
    d3 = df[df['total'] >= 20]
    rows.append(row('WLS, only n >= 20', fit(d3, BASE, d3['total'].values), BASE, len(d3)))
    print(pd.DataFrame(rows).to_string(index=False))

    correct = np.round(df['uuas'] * df['total'])
    model = sm.GLM(np.column_stack([correct, df['total'] - correct]),
                   sm.add_constant(df[BASE].values), family=sm.families.Binomial())
    # Quasi-binomial: dispersion = Pearson X2 / df, passed explicitly because
    # fit(scale='X2') mis-estimates it with a two-column (successes, failures) endog.
    phi = model.fit().pearson_chi2 / (len(df) - len(BASE) - 1)
    glm = model.fit(scale=phi)
    print('\nquasi-binomial GLM, logit link, dispersion = Pearson X2 / df:')
    for i, pname in enumerate(BASE, start=1):
        print(f'  {pname:18s} {glm.params[i]:+.3f}  (p={glm.pvalues[i]:.3g})')
    print(f'  dispersion {glm.scale:.2f}')

    # WLS is OLS on sqrt(w)-scaled data; Cook's distance is taken there.
    sw = np.sqrt(w)
    ols = sm.OLS(df['uuas'].values * sw, sm.add_constant(df[BASE].values) * sw[:, None]).fit()
    infl = ols.get_influence().cooks_distance[0]
    top = pd.Series(infl, index=df.index).nlargest(5)
    print('\nlargest Cook\'s distances (4/n =', round(4 / len(df), 3), '):',
          ', '.join(f'{k} {v:.2f}' for k, v in top.items()))
    loo = []
    for rel in df.index:
        d = df.drop(rel)
        loo.append((rel, fit(d, BASE, d['total'].values).params[1]))
    loo = pd.Series(dict(loo))
    print(f'leave-one-relation-out beta_ent: min {loo.min():+.3f} ({loo.idxmin()}), '
          f'max {loo.max():+.3f} ({loo.idxmax()})')


def block_dependent(df):
    print('\n### dependent entropy (r12d W3)')
    w = df['total'].values
    preds = BASE + ['dep_sim_entropy']
    r = np.corrcoef(df['head_sim_entropy'], df['dep_sim_entropy'])[0, 1]
    print(f'r(head_sim_entropy, dep_sim_entropy) over {len(df)} relations = {r:+.2f}')
    rows = [row('published', fit(df, BASE, w), BASE, len(df)),
            row('+ dependent sim-entropy', fit(df, preds, w), preds, len(df)),
            row('dependent instead of head',
                fit(df, ['dep_sim_entropy', 'mean_log_length', 'sd_log_length'], w),
                ['dep_sim_entropy', 'mean_log_length', 'sd_log_length'], len(df))]
    print(pd.DataFrame(rows).to_string(index=False))
    print('VIF with both:', vif_table(df, preds))


def block_confounds(df):
    print('\n### confounds (U2Ge Q3)')
    w = df['total'].values
    extras = ['log_train_freq', 'head_shannon_entropy', 'log_n_head_types',
              'head_pos_entropy', 'verb_head_share']
    print('correlation of each control with head_sim_entropy:',
          {e: round(np.corrcoef(df[e], df['head_sim_entropy'])[0, 1], 2) for e in extras})
    rows = [row('published', fit(df, BASE, w), BASE, len(df))]
    for e in extras:
        preds = BASE + [e]
        m = fit(df, preds, w)
        r = row(f'+ {e}', m, BASE, len(df))
        r['control'] = f'{m.params[-1]:+.3f} (p={m.pvalues[-1]:.3g})'
        rows.append(r)
    preds = BASE + extras
    m = fit(df, preds, w)
    r = row('+ all five', m, BASE, len(df))
    rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print('all-five model, controls:',
          {e: f'{m.params[i]:+.3f} (p={m.pvalues[i]:.3g})'
           for i, e in enumerate(preds, start=1) if e in extras})
    print('VIF, all-five model:', vif_table(df, preds))
    # Nested F-test: does head_sim_entropy add to a model with every control?
    no_ent = [p for p in preds if p != 'head_sim_entropy']
    m0 = fit(df, no_ent, w)
    f, pval, _ = m.compare_f_test(m0)
    print(f'head_sim_entropy over all controls: F = {f:.2f}, p = {pval:.3g}, '
          f'partial dR2 = {m.rsquared - m0.rsquared:.3f}')


def block_alt_sim(args):
    print('\n### alternative similarity spaces (U2Ge Q4, r12d Q2)')
    rows = []
    for spec in [f'published={args.sim}'] + (args.alt_sim or []):
        label, path = spec.split('=', 1)
        df = load_regression_frame(args.uuas, path, args.len, columns=['uuas', 'total'] + BASE)
        rows.append(row(label, fit(df, BASE, df['total'].values), BASE, len(df)))
    print(pd.DataFrame(rows).to_string(index=False))


def block_replicates(args):
    files = sorted(glob.glob(args.replicates))
    print(f'\n### replicates ({len(files)} probing runs: {args.replicates})')
    coefs, r2 = [], []
    for f in files:
        df = load_regression_frame(f, args.sim, args.len, columns=['uuas', 'total'] + BASE)
        m = fit(df, BASE, df['total'].values)
        coefs.append(list(m.params[1:]) + list(m.pvalues[1:]))
        r2.append(m.rsquared)
    c = pd.DataFrame(coefs, columns=BASE + [f'p_{b}' for b in BASE])
    for b in BASE:
        print(f'  {b:18s} mean {c[b].mean():+.3f}  sd {c[b].std():.3f}  '
              f'range [{c[b].min():+.3f}, {c[b].max():+.3f}]  '
              f'same sign {int((np.sign(c[b]) == np.sign(c[b].iloc[0])).sum())}/{len(c)}  '
              f'p<.05 {int((c[f"p_{b}"] < .05).sum())}/{len(c)}  max p {c[f"p_{b}"].max():.3g}')
    print(f'  R2 mean {np.mean(r2):.3f}  range [{min(r2):.3f}, {max(r2):.3f}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uuas', required=True)
    ap.add_argument('--sim', required=True)
    ap.add_argument('--len', required=True)
    ap.add_argument('--conllx', required=True, help='PTB train, for the POS controls')
    ap.add_argument('--alt-sim', action='append')
    ap.add_argument('--replicates')
    ap.add_argument('--blocks', default='weights,dependent,confounds,alt-sim,replicates')
    args = ap.parse_args()
    blocks = args.blocks.split(',')

    df = load_regression_frame(args.uuas, args.sim, args.len, columns=['uuas', 'total'] + BASE)
    df = with_extras(df, args.sim, args.len, args.conllx)
    print(f'{len(df)} relations; UASL from {args.uuas}')
    if 'weights' in blocks:
        block_weights(df)
    if 'dependent' in blocks:
        block_dependent(df)
    if 'confounds' in blocks:
        block_confounds(df)
    if 'alt-sim' in blocks:
        block_alt_sim(args)
    if 'replicates' in blocks and args.replicates:
        block_replicates(args)


if __name__ == '__main__':
    main()
