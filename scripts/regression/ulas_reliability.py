#!/usr/bin/env python3
"""How much of the between-relation variance in ULAS is real, not sampling noise?

A relation's ULAS is a proportion estimated from a finite number of dev-set
edges, so part of the spread of ULAS across relations is binomial sampling
noise rather than genuine between-relation difference. No regression can explain
that part, so it sets a ceiling on the attainable R^2:

    Var_obs  =  Var_true + E[ p(1-p)/n ]
    reliability  =  Var_true / Var_obs   <=  max attainable R^2

with all moments taken under the same weights the regression uses. This matters
when comparing a model whose ULAS is high and well spread (BERT-base) with one
whose ULAS sits near a floor (a model pre-trained on permuted sentences): a low
R^2 for the second could in principle mean 'nothing left to explain' rather than
'these predictors do not apply'.

Reported per model so the comparison is explicit. Note that sampling noise
attenuates coefficients toward zero but cannot reverse their sign, so a sign
change is evidence that noise alone does not explain.

Usage:
    python ulas_reliability.py --spec "BERT-base:RESULTS:16" \\
        --spec "RoBERTa-Shuffle-n1:RESULTS2:9"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load(uuas_path):
    u = pd.read_csv(uuas_path, sep='\t')
    if 'total' not in u.columns and {'correct', 'uuas'}.issubset(u.columns):
        u['total'] = u['correct'] / u['uuas']
    return u.dropna(subset=['uuas', 'total'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', action='append', required=True,
                    help='LABEL:RESULTS_DIR:CHECKPOINT (repeatable)')
    args = ap.parse_args()

    print(f'{"model":22s} {"n":>4s} {"mean":>7s} {"sd_obs":>8s} {"sd_noise":>9s} '
          f'{"sd_true":>8s} {"reliability":>12s}')
    for spec in args.spec:
        label, results, ck = spec.rsplit(':', 2)
        df = load(Path(results) / f'layer-{int(ck):02d}' / 'dev.uuas_by_relation')
        p = df['uuas'].to_numpy(dtype=float)
        n = df['total'].to_numpy(dtype=float)
        w = n / n.sum()

        mean = float(np.sum(w * p))
        var_obs = float(np.sum(w * (p - mean) ** 2))
        # Binomial sampling variance of each relation's own estimate.
        var_noise = float(np.sum(w * p * (1.0 - p) / np.maximum(n, 1.0)))
        var_true = max(var_obs - var_noise, 0.0)
        rel = var_true / var_obs if var_obs > 0 else float('nan')
        print(f'{label:22s} {len(df):4d} {mean:7.4f} {np.sqrt(var_obs):8.4f} '
              f'{np.sqrt(var_noise):9.4f} {np.sqrt(var_true):8.4f} {rel:12.3f}')


if __name__ == '__main__':
    main()
