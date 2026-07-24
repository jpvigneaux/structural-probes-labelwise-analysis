#!/usr/bin/env python3
"""
Three-panel single-column PNG analogous to the upper panel of
uuas_mean_curves_by_checkpoint.png, but with the mean taken over
three curated relation subsets from hierarchical clustering.

For each (subset, checkpoint, distance n):
  - linearly interpolate each relation's UAS curve within its observed range
  - aggregate over relations in the subset that cover n  (no extrapolation)
  - require at least MIN_REL relations to contribute before drawing a point

Usage (login node):
    python uuas_mean_curves_by_cluster_png.py \
        [--curves figures/uuas_mean_curves_by_checkpoint.npz] \
        [--out    figures/uuas_mean_curves_by_cluster.png]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mcm


# ── Curated subsets ───────────────────────────────────────────────────────────

SUBSETS = [
    ('det, amod, prt, nsubj, tmod',
     ['det', 'amod', 'prt', 'nsubj', 'tmod']),
    ('prep, cop, iobj, advmod, pcomp, mark',
     ['prep', 'cop', 'iobj', 'advmod', 'pcomp', 'mark']),
    ('pobj, dobj, auxpass, acomp, npadvmod, xcomp',
     ['pobj', 'dobj', 'auxpass', 'acomp', 'npadvmod', 'xcomp']),
]

MIN_REL = 2          # min relations covering n to draw the point


# ── Core computation ──────────────────────────────────────────────────────────

def compute_subset_curves(npz, ci, subset_rels, rel_to_ri):
    """
    For one checkpoint index ci and one relation subset, return
    (xs, means, stds) after interpolation + aggregation.
    """
    # Build per-relation (ns, uas) lookup
    rel_data = {}
    for rel in subset_rels:
        if rel not in rel_to_ri:
            continue
        ri  = rel_to_ri[rel]
        ns  = npz[f'rel_ns_{ci}_{ri}'].astype(float)
        uas = npz[f'rel_uas_{ci}_{ri}']
        if len(ns) >= 1:
            rel_data[rel] = (ns, uas)

    if not rel_data:
        return np.array([]), np.array([]), np.array([])

    # Union of all covered integer x-values
    covered = set()
    for ns, _ in rel_data.values():
        covered.update(range(int(ns[0]), int(ns[-1]) + 1))

    xs_out, means_out, stds_out = [], [], []
    for n in sorted(covered):
        vals = [
            float(np.interp(n, ns, uas))
            for ns, uas in rel_data.values()
            if ns[0] <= n <= ns[-1]
        ]
        if len(vals) >= MIN_REL:
            xs_out.append(n)
            means_out.append(float(np.mean(vals)))
            stds_out.append(float(np.std(vals, ddof=0)))

    return np.array(xs_out), np.array(means_out), np.array(stds_out)


# ── Plotting helper ───────────────────────────────────────────────────────────

def plot_curve(ax, xs, means, stds, color, label=None):
    if len(xs) == 0:
        return None
    upper = means + stds
    lower = np.maximum(means - stds, 0.0)
    ax.fill_between(xs, lower, upper, alpha=0.15, color=color, linewidth=0)
    line, = ax.plot(xs, means, '-o', color=color, linewidth=1.2,
                    markersize=3, label=label)
    return line


def style_ax(ax, log_x=False):
    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=(-0.05 if log_x else -0.3))
    if not log_x:
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', linewidth=0.4, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Mean labeled UAS', fontsize=8)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).parent
    p.add_argument('--curves',
                   default=str(script_dir / 'figures'
                               / 'uuas_mean_curves_by_checkpoint.npz'))
    p.add_argument('--out', default=None)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--log-x', action='store_true',
                   help='Plot ln(n+1) on the x-axis instead of n')
    return p.parse_args()


def main():
    args = parse_args()
    npz  = np.load(args.curves, allow_pickle=False)
    suffix = 'uuas_mean_curves_by_cluster_logn.png' if args.log_x \
             else 'uuas_mean_curves_by_cluster.png'
    out_path = Path(args.out) if args.out \
               else Path(args.curves).parent / suffix

    all_checkpoints = npz['checkpoints'].tolist()
    relations       = npz['rel_names'].astype(str).tolist()
    rel_to_ri       = {r: i for i, r in enumerate(relations)}

    # Use only the odd checkpoints (3,5,...,25) — same as the upper panel
    main_cks = [ck for ck in all_checkpoints if ck % 2 == 1]
    ck_to_ci = {ck: i for i, ck in enumerate(all_checkpoints)}

    # Warn about any unknown relation names in the subsets
    for _, rels in SUBSETS:
        for r in rels:
            if r not in rel_to_ri:
                print(f'  WARNING: relation "{r}" not found in NPZ, skipping')

    # ── Colours: Viridis over main_cks ───────────────────────────────────────
    cmap   = mcm.get_cmap('viridis')
    t_vals = np.linspace(0.05, 0.92, len(main_cks))
    colors = [cmap(t) for t in t_vals]

    # ── Figure: three stacked panels sharing x-axis ───────────────────────────
    n_panels = len(SUBSETS)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(3.5, 2.4 * n_panels + 0.9),
                             sharex=True)

    handles = None
    for ax, (title, subset_rels) in zip(axes, SUBSETS):
        h_list = []
        for ck, color in zip(main_cks, colors):
            ci = ck_to_ci[ck]
            xs, means, stds = compute_subset_curves(npz, ci, subset_rels, rel_to_ri)
            if len(xs) > 0 and args.log_x:
                xs = np.log(xs + 1)
            h = plot_curve(ax, xs, means, stds, color, label=str(ck))
            if h is not None:
                h_list.append(h)

        style_ax(ax, args.log_x)
        ax.set_title(title, fontsize=7.5, pad=3)

        if handles is None and h_list:
            handles = h_list    # collect once for the shared legend

    xlabel = r'$\ln(n+1)$' if args.log_x else '$n$ (words between endpoints)'
    axes[-1].set_xlabel(xlabel, fontsize=8)
    # Hide x-tick labels on all but the last panel
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    # ── Shared legend below all panels ────────────────────────────────────────
    fig.legend(
        handles=handles,
        labels=[str(ck) for ck in main_cks],
        title='Checkpoint',
        title_fontsize=7,
        fontsize=7,
        loc='lower center',
        ncol=6,
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        frameon=True,
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.12, hspace=0.30)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
