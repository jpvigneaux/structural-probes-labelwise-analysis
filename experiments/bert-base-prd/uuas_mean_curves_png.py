#!/usr/bin/env python3
"""
Two-panel single-column PNG of the mean UAS-by-distance curves.

Top panel   : checkpoints 3, 5, 7, …, 25  (post-block residuals)
Bottom panel: checkpoint 16 only           (optimal probe)

Loads pre-computed curve data from the NPZ saved by
uuas_mean_curves_by_checkpoint.py.

Usage (login node, fast):
    python uuas_mean_curves_png.py \
        [--curves figures/uuas_mean_curves_by_checkpoint.npz] \
        [--out    figures/uuas_mean_curves_by_checkpoint.png] \
        [--dpi    300]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mcm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    script_dir = Path(__file__).parent
    p.add_argument('--curves',
                   default=str(script_dir / 'figures' / 'uuas_mean_curves_by_checkpoint.npz'),
                   help='NPZ file written by uuas_mean_curves_by_checkpoint.py')
    p.add_argument('--out', default=None,
                   help='Output PNG path (default: same stem as --curves)')
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--log-x', action='store_true',
                   help='Plot ln(n+1) on the x-axis instead of n')
    return p.parse_args()


def transform_x(xs, log_x):
    return np.log(xs + 1) if log_x else xs


def plot_curve(ax, xs, means, stds, color, label=None):
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


def main():
    args = parse_args()
    npz  = np.load(args.curves)
    base = Path(args.curves).with_suffix('')
    out_path = (
        Path(args.out) if args.out
        else Path(str(base) + ('_logn.png' if args.log_x else '.png'))
    )

    all_checkpoints = npz['checkpoints'].tolist()

    # Split into the two sets
    main_cks = [ck for ck in all_checkpoints if ck != 16 and ck % 2 == 1]
    ref_ck   = 16

    if ref_ck not in all_checkpoints:
        raise ValueError(
            f'Checkpoint {ref_ck} not found in NPZ. '
            'Re-run the SLURM job with --checkpoints including 16.'
        )

    # Index lookup
    ck_to_ci = {ck: ci for ci, ck in enumerate(all_checkpoints)}

    def get_curve(ck):
        ci = ck_to_ci[ck]
        return (npz[f'xs_{ci}'], npz[f'means_{ci}'],
                npz[f'stds_{ci}'],  npz[f'nrels_{ci}'])

    # ── Colours: Viridis over main_cks only (same as interactive HTML) ────────
    cmap   = mcm.get_cmap('viridis')
    t_vals = np.linspace(0.05, 0.92, len(main_cks))
    colors = {ck: cmap(t) for ck, t in zip(main_cks, t_vals)}

    # Checkpoint 16 colour: viridis at its interpolated position in the full
    # odd sequence (between 15 and 17, so roughly the midpoint of those two)
    t_15 = t_vals[main_cks.index(15)]
    t_17 = t_vals[main_cks.index(17)]
    color_16 = cmap((t_15 + t_17) / 2)

    # ── Figure: two stacked panels sharing x-axis ─────────────────────────────
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.5, 5.4), sharex=True,
        gridspec_kw={'height_ratios': [1.6, 1]},
    )

    # — Top panel —
    handles = []
    for ck in main_cks:
        xs, means, stds, _ = get_curve(ck)
        if len(xs) == 0:
            continue
        h = plot_curve(ax_top, transform_x(xs, args.log_x), means, stds,
                       colors[ck], label=str(ck))
        handles.append(h)

    style_ax(ax_top, args.log_x)
    ax_top.set_title('Post-block checkpoints (3 – 25)', fontsize=8, pad=4)
    ax_top.tick_params(labelbottom=False)   # x labels hidden (shared axis)

    # — Bottom panel —
    xs16, means16, stds16, _ = get_curve(ref_ck)
    plot_curve(ax_bot, transform_x(xs16, args.log_x), means16, stds16,
               color_16, label='16')

    style_ax(ax_bot, args.log_x)
    xlabel = r'$\ln(n+1)$' if args.log_x else '$n$ (words between endpoints)'
    ax_bot.set_xlabel(xlabel, fontsize=8)
    ax_bot.set_title('Optimal checkpoint (16)', fontsize=8, pad=4)

    # ── Legend below both panels, 6 columns ───────────────────────────────────
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

    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=0.20, hspace=0.12)

    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
