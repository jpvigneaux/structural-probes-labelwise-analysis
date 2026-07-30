#!/usr/bin/env python3
"""Diagnostics for the choice of range component in the composite distance.

The composite distance of the clustering analysis is

    d(r1,r2) = sqrt( alpha * d_ULAS_std^2 + (1-alpha) * d_range_std^2 ),

and the question this script addresses is whether replacing the published
`d_range` -- the absolute difference of count-weighted 90th-percentile arc
lengths -- by the Wasserstein-1 distance between the full arc-length
distributions produces more interpretable clusters at intermediate alpha.

"Interpretable" is not directly measurable, so three proxies are reported, all
of which are properties of the clustering rather than of a reading of it:

  cophenetic r  How faithfully the dendrogram's merge heights reproduce the
                pairwise distances. A low value means the tree is a poor
                summary of the metric, so any reading of it is unreliable.
  singletons    Relations left alone at the 25%-of-tallest-merge cut. A
                partition that is mostly singletons plus one large residue
                carries little structure to interpret.
  ARI vs a=1    Adjusted Rand index of the partition against the pure-ULAS
                partition. Near 1 means the composite adds nothing to alpha=1;
                near 0 means the intermediate solution is genuinely its own
                grouping rather than a perturbation of one endpoint.

Also reported: the correlation of the two range metrics with each other and of
each with d_ULAS, since a range component nearly orthogonal to d_ULAS is what
makes intermediate alpha hard to read in the first place.

Usage:
    python compare_range_metrics.py --curves uuas_mean_curves_bertbase.npz
"""

import argparse

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from cluster_relations_by_distance import (
    compute_dist_uas, compute_dist_range, compute_dist_wasserstein,
    standardise, weighted_percentile,
)

ALPHAS = [0.0, 0.33, 0.66, 1.0]


def adjusted_rand(a, b):
    """Adjusted Rand index between two flat labellings."""
    a, b = np.asarray(a), np.asarray(b)
    ua, ub = np.unique(a), np.unique(b)
    n = len(a)
    cont = np.zeros((len(ua), len(ub)), dtype=float)
    for i, x in enumerate(ua):
        for j, y in enumerate(ub):
            cont[i, j] = np.sum((a == x) & (b == y))

    def c2(v):
        return v * (v - 1) / 2.0

    sum_ij = c2(cont).sum()
    sum_i = c2(cont.sum(axis=1)).sum()
    sum_j = c2(cont.sum(axis=0)).sum()
    expected = sum_i * sum_j / c2(n)
    maximum = 0.5 * (sum_i + sum_j)
    return (sum_ij - expected) / (maximum - expected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curves', required=True)
    ap.add_argument('--pct', type=float, default=90)
    ap.add_argument('--min-overlap', type=int, default=2)
    args = ap.parse_args()

    npz = np.load(args.curves, allow_pickle=False)
    relations = npz['rel_names'].astype(str).tolist()
    n_rel = len(relations)
    n_ck = len(npz['checkpoints'])

    d_ulas = compute_dist_uas(npz, n_rel, n_ck, args.min_overlap)
    n_pct = np.array([weighted_percentile(npz[f'rel_all_ns_{ri}'],
                                          npz[f'rel_counts_{ri}'], args.pct)
                      for ri in range(n_rel)])
    d_p90 = compute_dist_range(n_pct, n_rel, float(np.nanmax(n_pct)))
    d_w1 = compute_dist_wasserstein(npz, n_rel)

    iu = np.triu_indices(n_rel, 1)
    u_ulas, u_p90, u_w1 = d_ulas[iu], d_p90[iu], d_w1[iu]
    print(f'{n_rel} relations, {len(u_ulas)} pairs\n')
    print('component correlations (Spearman):')
    print(f'  p90  vs  w1     {spearmanr(u_p90, u_w1).statistic:+.3f}')
    print(f'  p90  vs  d_ULAS {spearmanr(u_p90, u_ulas).statistic:+.3f}')
    print(f'  w1   vs  d_ULAS {spearmanr(u_w1, u_ulas).statistic:+.3f}')

    su = standardise(d_ulas)
    ranges = {'p90': standardise(d_p90), 'w1': standardise(d_w1)}

    ref = {}
    for name, sr in ranges.items():
        # alpha = 1 is the same tree for both metrics; compute it once per name
        # so the ARI baseline is exactly the published pure-ULAS partition.
        d = np.sqrt(su ** 2)
        np.fill_diagonal(d, 0.0)
        Z = linkage(squareform(d), method='average')
        ref[name] = fcluster(Z, t=0.25 * float(Z[-1, 2]), criterion='distance')

    print(f'\n{"metric":6s} {"alpha":>6s} {"coph r":>8s} {"clusters":>9s} '
          f'{"singl":>6s} {"largest":>8s} {"ARI vs a=1":>11s}')
    for name, sr in ranges.items():
        for alpha in ALPHAS:
            d = np.sqrt(alpha * su ** 2 + (1 - alpha) * sr ** 2)
            np.fill_diagonal(d, 0.0)
            cond = squareform(d)
            Z = linkage(cond, method='average')
            coph = cophenet(Z, cond)[0]
            lab = fcluster(Z, t=0.25 * float(Z[-1, 2]), criterion='distance')
            sizes = np.bincount(lab)[1:]
            print(f'{name:6s} {alpha:6.2f} {coph:8.3f} {len(sizes):9d} '
                  f'{int((sizes == 1).sum()):6d} {int(sizes.max()):8d} '
                  f'{adjusted_rand(lab, ref[name]):11.3f}')

    # Which relations the two range metrics disagree about most: rank each
    # relation by its mean distance to all others under each metric.
    mean_p90 = (d_p90.sum(axis=1) / (n_rel - 1))
    mean_w1 = (d_w1.sum(axis=1) / (n_rel - 1))
    rank_p90 = np.argsort(np.argsort(-mean_p90))
    rank_w1 = np.argsort(np.argsort(-mean_w1))
    shift = rank_w1 - rank_p90
    order = np.argsort(-np.abs(shift))
    print('\nrelations whose peripherality changes most between the metrics')
    print('  (rank 0 = most distant from the rest; positive shift = W1 moves it inward)')
    for ri in order[:8]:
        print(f'  {relations[ri]:12s} p90 rank {rank_p90[ri]:2d} -> '
              f'w1 rank {rank_w1[ri]:2d}  ({shift[ri]:+d})')


if __name__ == '__main__':
    main()
