#!/usr/bin/env python3
"""
Compute d_UAS and d_range for all relation pairs and report their
value distributions, to calibrate normalization before combining them
into the composite clustering distance.

Usage (login node):
    python inspect_distance_ranges.py \
        [--curves figures/uuas_mean_curves_by_checkpoint.npz] \
        [--pct 90]
"""
import argparse
from pathlib import Path

import numpy as np


def weighted_percentile(ns, counts, p):
    total = counts.sum()
    if total == 0:
        return np.nan
    cumulative = np.cumsum(counts)
    idx = min(int(np.searchsorted(cumulative, p / 100.0 * total)), len(ns) - 1)
    return float(ns[idx])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).parent
    p.add_argument('--curves',
                   default=str(script_dir / 'figures'
                               / 'uuas_mean_curves_by_checkpoint.npz'))
    p.add_argument('--pct', type=float, default=90)
    p.add_argument('--min-overlap', type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    npz  = np.load(args.curves, allow_pickle=False)
    checkpoints = npz['checkpoints'].tolist()
    relations   = npz['rel_names'].astype(str).tolist()
    n_ck  = len(checkpoints)
    n_rel = len(relations)

    # ── Range distances ───────────────────────────────────────────────────────
    n_pct = np.array([
        weighted_percentile(npz[f'rel_all_ns_{ri}'],
                            npz[f'rel_counts_{ri}'], args.pct)
        for ri in range(n_rel)
    ])
    n_max = float(np.nanmax(n_pct))

    dist_range = np.zeros((n_rel, n_rel))
    for i in range(n_rel):
        for j in range(i + 1, n_rel):
            p1, p2 = n_pct[i], n_pct[j]
            dr = abs(p1 - p2) / n_max \
                 if not (np.isnan(p1) or np.isnan(p2)) else 1.0
            dist_range[i, j] = dist_range[j, i] = dr

    # ── UAS distances (two-level) ─────────────────────────────────────────────
    dist_uas = np.ones((n_rel, n_rel))
    np.fill_diagonal(dist_uas, 0.0)
    for i in range(n_rel):
        for j in range(i + 1, n_rel):
            per_ck_mse = []
            for ci in range(n_ck):
                ns1  = npz[f'rel_ns_{ci}_{i}'].astype(float)
                uas1 = npz[f'rel_uas_{ci}_{i}']
                ns2  = npz[f'rel_ns_{ci}_{j}'].astype(float)
                uas2 = npz[f'rel_uas_{ci}_{j}']
                if len(ns1) < 1 or len(ns2) < 1:
                    continue
                xlo = int(max(ns1[0], ns2[0]))
                xhi = int(min(ns1[-1], ns2[-1]))
                if xhi - xlo + 1 < args.min_overlap:
                    continue
                xs = np.arange(xlo, xhi + 1, dtype=float)
                y1 = np.interp(xs, ns1, uas1)
                y2 = np.interp(xs, ns2, uas2)
                per_ck_mse.append(float(np.mean((y1 - y2) ** 2)))
            if per_ck_mse:
                dist_uas[i, j] = dist_uas[j, i] = float(np.sqrt(np.mean(per_ck_mse)))

    # ── Report ────────────────────────────────────────────────────────────────
    upper_uas   = np.array([dist_uas[i,j]   for i in range(n_rel) for j in range(i+1,n_rel)])
    upper_range = np.array([dist_range[i,j] for i in range(n_rel) for j in range(i+1,n_rel)])

    def stats(label, v):
        print(f'{label}')
        print(f'  mean   = {v.mean():.4f}')
        print(f'  std    = {v.std():.4f}')
        print(f'  median = {np.median(v):.4f}')
        print(f'  p25    = {np.percentile(v, 25):.4f}')
        print(f'  p75    = {np.percentile(v, 75):.4f}')
        print(f'  max    = {v.max():.4f}')

    print(f'Pairwise distance statistics ({n_rel*(n_rel-1)//2} pairs)\n')
    stats('d_UAS  (two-level, intersection-based RMS):', upper_uas)
    print()
    stats(f'd_range (|n_p90(r1)-n_p90(r2)| / {n_max:.0f}):', upper_range)

    ratio = upper_uas.std() / upper_range.std() if upper_range.std() > 0 else np.nan
    print(f'\nstd(d_UAS) / std(d_range) = {ratio:.3f}')
    print(f'=> to equalise std, scale d_range by {ratio:.3f} before combining')


if __name__ == '__main__':
    main()
