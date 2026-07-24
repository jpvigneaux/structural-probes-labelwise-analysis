#!/usr/bin/env python3
"""
Cluster PTB dependency relations by the similarity of their UAS-by-distance
curves, using data from all available checkpoints simultaneously.

Distance metric
---------------
A composite of two orthogonal components, each standardised to unit std
across all pairwise distances before combining, so that alpha directly
controls the relative contribution:

  d(r1, r2)  =  sqrt( alpha * d_UAS_std(r1,r2)^2
                     + (1-alpha) * d_range_std(r1,r2)^2 )

UAS component (probe property)
  d_UAS: two-level average — per-checkpoint MSE first, then mean over
  checkpoints — so every checkpoint contributes equally regardless of
  how many shared distance values it has.

Range component (linguistic / corpus property)
  d_range = |n_p(r1) - n_p(r2)| / N_max
  where n_p(r) is the count-weighted p-th percentile of the arc-length
  distribution for relation r (same across checkpoints — corpus property).
  N_max = max_r n_p(r).

Default output: a 4-panel figure with alpha = 0, 0.33, 0.66, 1.
Pass --alpha to get a single-panel figure instead.

Usage (login node):
    python cluster_relations_by_distance.py \\
        [--curves  figures/uuas_mean_curves_by_checkpoint.npz] \\
        [--out     figures/relation_clustering_dendrogram.png]  \\
        [--alpha   FLOAT]  # single panel; omit for 4-panel figure
        [--pct     90]
        [--min-overlap 2]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform


# ── Helpers ───────────────────────────────────────────────────────────────────

def weighted_percentile(ns, counts, p):
    total = counts.sum()
    if total == 0:
        return np.nan
    idx = min(int(np.searchsorted(np.cumsum(counts), p / 100.0 * total)),
              len(ns) - 1)
    return float(ns[idx])


def compute_dist_uas(npz, n_rel, n_ck, min_overlap):
    """Two-level intersection-based RMS distance matrix."""
    dist = np.ones((n_rel, n_rel))
    np.fill_diagonal(dist, 0.0)
    for i in range(n_rel):
        for j in range(i + 1, n_rel):
            per_ck = []
            for ci in range(n_ck):
                ns1  = npz[f'rel_ns_{ci}_{i}'].astype(float)
                uas1 = npz[f'rel_uas_{ci}_{i}']
                ns2  = npz[f'rel_ns_{ci}_{j}'].astype(float)
                uas2 = npz[f'rel_uas_{ci}_{j}']
                if len(ns1) < 1 or len(ns2) < 1:
                    continue
                xlo = int(max(ns1[0], ns2[0]))
                xhi = int(min(ns1[-1], ns2[-1]))
                if xhi - xlo + 1 < min_overlap:
                    continue
                xs = np.arange(xlo, xhi + 1, dtype=float)
                per_ck.append(float(np.mean(
                    (np.interp(xs, ns1, uas1) - np.interp(xs, ns2, uas2)) ** 2
                )))
            if per_ck:
                dist[i, j] = dist[j, i] = float(np.sqrt(np.mean(per_ck)))
    return dist


def compute_dist_range(n_pct, n_rel, n_max):
    """Normalised absolute difference of count-weighted percentile arc lengths."""
    dist = np.zeros((n_rel, n_rel))
    for i in range(n_rel):
        for j in range(i + 1, n_rel):
            p1, p2 = n_pct[i], n_pct[j]
            dr = abs(p1 - p2) / n_max \
                 if not (np.isnan(p1) or np.isnan(p2)) else 1.0
            dist[i, j] = dist[j, i] = dr
    return dist


def standardise(dist_mat):
    """Divide upper-triangle values by their std so the component has unit std."""
    n = dist_mat.shape[0]
    upper = np.array([dist_mat[i, j] for i in range(n) for j in range(i+1, n)])
    s = upper.std()
    if s == 0:
        return dist_mat.copy()
    return dist_mat / s


def draw_dendrogram(ax, Z, labels, title, alpha_val):
    color_thresh = 0.25 * float(Z[-1, 2])
    ddata = dendrogram(Z, labels=labels, orientation='left', ax=ax,
                       leaf_font_size=11, color_threshold=color_thresh)
    ax.set_xscale('symlog', linthresh=0.3, linscale=0.5)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda x, _: f'{x:g}' if x < 1 else f'{x:.0f}'
    ))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.axvline(color_thresh, color='gray', linewidth=0.9, linestyle='--')
    ax.set_xlabel('Avg composite distance (symlog)', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.set_title(title, fontsize=12, pad=6)

    # Colour each leaf label to match its cluster colour
    leaf_colors = ddata['leaves_color_list']   # one entry per leaf, bottom→top
    for tick, color in zip(ax.get_yticklabels(), leaf_colors):
        tick.set_color(color)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).parent
    p.add_argument('--curves',
                   default=str(script_dir / 'figures'
                               / 'uuas_mean_curves_by_checkpoint.npz'))
    p.add_argument('--out', default=None,
                   help='Output PNG (default: figures/relation_clustering_dendrogram.png'
                        ' for single-alpha, or figures/relation_clustering_4panel.png)')
    p.add_argument('--alpha', type=float, default=None,
                   help='Single alpha value; omit to produce 4-panel figure')
    p.add_argument('--pct', type=float, default=90)
    p.add_argument('--min-overlap', type=int, default=2)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args      = parse_args()
    npz       = np.load(args.curves, allow_pickle=False)
    relations = npz['rel_names'].astype(str).tolist()
    n_rel     = len(relations)
    n_ck      = len(npz['checkpoints'])

    # ── Compute raw distance components ──────────────────────────────────────
    print('Computing d_UAS …')
    dist_uas = compute_dist_uas(npz, n_rel, n_ck, args.min_overlap)

    print('Computing d_range …')
    n_pct = np.array([weighted_percentile(npz[f'rel_all_ns_{ri}'],
                                          npz[f'rel_counts_{ri}'], args.pct)
                      for ri in range(n_rel)])
    n_max = float(np.nanmax(n_pct))
    dist_range = compute_dist_range(n_pct, n_rel, n_max)

    # ── Standardise to unit std ───────────────────────────────────────────────
    dist_uas_std   = standardise(dist_uas)
    dist_range_std = standardise(dist_range)

    upper_u = [dist_uas_std[i,j]   for i in range(n_rel) for j in range(i+1,n_rel)]
    upper_r = [dist_range_std[i,j] for i in range(n_rel) for j in range(i+1,n_rel)]
    print(f'After standardisation — d_UAS_std:   std={np.std(upper_u):.3f}  '
          f'max={np.max(upper_u):.3f}')
    print(f'After standardisation — d_range_std: std={np.std(upper_r):.3f}  '
          f'max={np.max(upper_r):.3f}')

    # ── Build dendrogram(s) ───────────────────────────────────────────────────
    alphas     = [args.alpha] if args.alpha is not None else [0.0, 0.33, 0.66, 1.0]
    single     = len(alphas) == 1
    out_stem   = ('relation_clustering_dendrogram'
                  if single else 'relation_clustering_4panel')
    out_path   = (Path(args.out) if args.out
                  else Path(args.curves).parent / f'{out_stem}.png')

    if single:
        fig, axes = plt.subplots(1, 1, figsize=(4.5, 11.0))
        axes = [axes]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10.0, 14.0))
        axes = axes.flatten()

    print('Clustering and plotting …')
    for ax, alpha in zip(axes, alphas):
        dist_comp = np.sqrt(alpha * dist_uas_std ** 2
                            + (1 - alpha) * dist_range_std ** 2)
        np.fill_diagonal(dist_comp, 0.0)

        Z = linkage(squareform(dist_comp), method='average')
        title = (f'α = {alpha:.2f}  '
                 f'({"UAS only" if alpha == 1.0 else "range only" if alpha == 0.0 else f"UAS {alpha:.0%} / range {1-alpha:.0%}"})')
        draw_dendrogram(ax, Z, relations, title, alpha)

        # Print flat clusters
        color_thresh = 0.25 * float(Z[-1, 2])
        labels_flat  = fcluster(Z, t=color_thresh, criterion='distance')
        clusters = {}
        for rel, cid in zip(relations, labels_flat):
            clusters.setdefault(cid, []).append(rel)
        print(f'\nα={alpha:.2f} — flat clusters (threshold={color_thresh:.3f}):')
        for cid, rels in sorted(clusters.items(), key=lambda x: -len(x[1])):
            print(f'  ({len(rels)}) {", ".join(sorted(rels))}')

    if single:
        fig.suptitle(
            f'Hierarchical clustering of dependency relations\n'
            f'(α={alphas[0]:.2f}, {args.pct:.0f}th-pct range, '
            f'all checkpoints, test set)',
            fontsize=12, y=1.01,
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
