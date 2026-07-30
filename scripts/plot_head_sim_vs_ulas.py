#!/usr/bin/env python3
"""Scatter of head similarity-corrected entropy against ULAS, per relation.

Reproduces the main-text entropy figure for any model, so the appendix can show
how model-specific the entropy/ULAS relationship is. Point area is proportional
to how many dev-set edges the relation's ULAS was measured from -- the same
quantity that weights the WLS regression -- so visually dominant points are also
the statistically reliable ones.

The fitted line is the weighted least-squares fit of ULAS on head sim-entropy
alone (the bivariate marginal), not the full two-predictor model; it is drawn to
show direction and spread, not to restate the regression.

Usage:
    python plot_head_sim_vs_ulas.py --uuas dev.uuas_by_relation \\
        --sim results_sim_ptb.tsv --out fig.png --model-label "BERT-base (ck 16)"
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Relations worth naming on the plot: the extremes on either axis carry the
# argument, and labelling all 42 would be unreadable.
LABEL_N = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--uuas', required=True)
    ap.add_argument('--sim', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-label', default='')
    args = ap.parse_args()

    uuas = pd.read_csv(args.uuas, sep='\t').set_index('relation')
    sim = pd.read_csv(args.sim, sep='\t').set_index('deprel')
    common = uuas.index.intersection(sim.index)
    if len(common) == 0:
        raise SystemExit('no relations in common between --uuas and --sim')

    df = pd.DataFrame({
        'uuas': uuas.loc[common, 'uuas'],
        'total': uuas.loc[common, 'total'],
        'ent': sim.loc[common, 'head_sim_entropy_bits'],
    }).dropna()

    w = df['total'].to_numpy(dtype=float)
    x = df['ent'].to_numpy(dtype=float)
    y = df['uuas'].to_numpy(dtype=float)

    # Weighted least squares fit of the marginal relationship.
    W = w / w.sum()
    xm, ym = np.sum(W * x), np.sum(W * y)
    slope = np.sum(W * (x - xm) * (y - ym)) / np.sum(W * (x - xm) ** 2)
    intercept = ym - slope * xm
    # Weighted correlation, for the annotation.
    r = (np.sum(W * (x - xm) * (y - ym))
         / np.sqrt(np.sum(W * (x - xm) ** 2) * np.sum(W * (y - ym) ** 2)))

    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(x, y, s=18 + 320 * (w / w.max()), alpha=0.55,
               color='#2a78d6', edgecolor='white', linewidth=0.8, zorder=3)

    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, intercept + slope * xs, color='#eb6834', linewidth=2, zorder=4,
            label=f'weighted fit (r = {r:.2f})')

    # Label the most frequent relations plus the axis extremes.
    to_label = set(df['total'].nlargest(LABEL_N).index) \
        | set(df['ent'].nlargest(3).index) | set(df['uuas'].nsmallest(3).index)
    for rel in to_label:
        ax.annotate(rel, (df.loc[rel, 'ent'], df.loc[rel, 'uuas']),
                    textcoords='offset points', xytext=(5, 4),
                    fontsize=7.5, color='#52514e')

    ax.set_xlabel('head similarity-corrected entropy (bits)', fontsize=11)
    ax.set_ylabel('ULAS', fontsize=11)
    if args.model_label:
        ax.set_title(args.model_label, fontsize=11.5, color='#0b0b0b')
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, color='#e1e0d9', linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc='lower left')

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'Saved {args.out}  (n={len(df)}, slope={slope:.4f}, weighted r={r:.3f})')


if __name__ == '__main__':
    main()
