#!/usr/bin/env python3
"""Per-relation UASL with error bars over probe replicates at one checkpoint.

For every relation: the UASL of the published probe, and the mean +/- one standard
deviation over (a) probes differing only in their random seed and (b) probes trained on
stratified re-partitions of the corpus. Relations are ordered by the published UASL.
Also prints the median per-relation standard deviation and the "weighted" standard
deviation sqrt(sum_r n_r var_r / sum_r n_r), n_r = published dev edge count (the
regression weight).

    python plot_replicate_uasl.py --published RES/layer-16/dev.uuas_by_relation \
        --seeds 'SEEDS/seed_*/layer-16/dev.uuas_by_relation' \
        --splits 'SPLITS/split_*/layer-16/dev.uuas_by_relation' --out fig.png
"""
import argparse
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read(path):
    t = pd.read_csv(path, sep='\t').set_index('relation')
    return t[t.index != 'UNK']


def stack(pattern, index):
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f'no files match {pattern}')
    return pd.concat([read(f)['uuas'].rename(f) for f in files], axis=1).reindex(index), len(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--published', required=True)
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--splits', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    pub = read(a.published)
    order = pub['uuas'].sort_values(ascending=False).index
    n = pub.loc[order, 'total'].astype(float)
    x = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(12, 3.6))
    for (label, pattern, dx, colour) in (('20 seeds', a.seeds, -0.18, '#1f77b4'),
                                         ('20 re-splits', a.splits, 0.18, '#d95f02')):
        M, k = stack(pattern, order)
        mean, sd = M.mean(axis=1), M.std(axis=1, ddof=1)
        w_sd = np.sqrt((n * sd ** 2).sum() / n.sum())
        print(f'{label}: {k} runs, median SD {sd.median():.4f}, weighted SD {w_sd:.4f}')
        ax.errorbar(x + dx, mean, yerr=sd, fmt='o', ms=3, lw=1, capsize=2,
                    color=colour, label=f'{label}: mean $\\pm$ 1 SD')
    ax.plot(x, pub.loc[order, 'uuas'], 'k_', ms=9, mew=1.5, label='published probe')
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=70, fontsize=8)
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel('dev UASL')
    ax.grid(axis='y', color='#dddddd', lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc='lower left', ncol=3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
