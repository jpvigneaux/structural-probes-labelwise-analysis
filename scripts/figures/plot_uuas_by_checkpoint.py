#!/usr/bin/env python3
"""UUAS by residual stream checkpoint for several runs, dev and test.

UUAS is pooled from the per-relation files (sum of correct over sum of total), which
equals the probe's own dev.uuas / test.uuas exactly; using the per-relation files lets a
checkpoint whose one-line summary was not written (a timed-out evaluation) still be shown.

    python plot_uuas_by_checkpoint.py --run "English (BERT-base)=RES_EN" \
        --run "Chinese (bert-base-chinese)=RES_ZH" --mark 16 --out fig.png
"""
import argparse
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def pooled(path):
    t = pd.read_csv(path, sep='\t')
    return t['correct'].sum() / t['total'].sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='append', required=True, help='LABEL=RESULTS_DIR')
    ap.add_argument('--mark', type=int, default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    colours = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd']
    for colour, spec in zip(colours, a.run):
        label, root = spec.split('=', 1)
        rows = []
        for d in sorted(glob.glob(os.path.join(root, 'layer-*'))):
            k = int(re.search(r'layer-(\d+)', d).group(1))
            rec = {'ck': k}
            for split in ('dev', 'test'):
                f = os.path.join(d, f'{split}.uuas_by_relation')
                rec[split] = pooled(f) if os.path.exists(f) else float('nan')
            rows.append(rec)
        df = pd.DataFrame(rows).sort_values('ck')
        best = df.loc[df['dev'].idxmax()]
        print(f'{label}: best dev checkpoint {int(best.ck)} dev {best.dev:.4f} test {best.test:.4f}')
        ax.plot(df.ck, df.dev, '-o', ms=2.5, color=colour, label=f'{label}, dev')
        ax.plot(df.ck, df.test, '--', lw=1, color=colour, label=f'{label}, test')
    if a.mark is not None:
        ax.axvline(a.mark, color='#888888', lw=0.8, ls=':')
    ax.set_xlabel('residual stream checkpoint')
    ax.set_ylabel('UUAS')
    ax.grid(color='#eeeeee', lw=0.6)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
