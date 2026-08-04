#!/usr/bin/env python3
"""
Two-panel single-column PNG of the mean UAS-by-distance curves.

Top panel   : every post-block checkpoint in the NPZ
Bottom panel: the optimal checkpoint alone (--ref-checkpoint)

Both panel titles are generated from the NPZ contents and --ref-checkpoint, so
they state the checkpoint set and optimal checkpoint of the model actually
plotted; these differ across the five models compared in the paper.

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
                   help='Plot ln(delta) on the x-axis instead of delta')
    p.add_argument('--ref-checkpoint', type=int, default=16,
                   help='Checkpoint highlighted in the lower panel: the optimal '
                        'probe for the model in question (BERT-base 16, '
                        'DeBERTa-v3-base 9, ModernBERT-base 30, GPT-2 16, GPT-J 8)')
    p.add_argument('--model-label', default='',
                   help='Model name for the panel titles')
    p.add_argument('--ref-desc', default='Optimal checkpoint',
                   help='How to describe the lower panel. The overall optimum of '
                        'a 2+2L model can be a post-attention index absent from '
                        'the curve set (GPT-2 16, ModernBERT-base 30), in which '
                        'case the best post-block checkpoint is shown instead and '
                        'the title should say so.')
    return p.parse_args()


def transform_x(xs, log_x):
    return np.log(xs) if log_x else xs


def describe_checkpoints(cks):
    """Compact description of a checkpoint set, e.g. '3, 5, …, 25' or '1–28'.

    Models stored in the 2+2L layout (BERT-base, ModernBERT-base, GPT-2) have
    post-block residuals at odd indices only, so their sets step by two; models
    stored as 1+L (DeBERTa-v3-base, GPT-J) have one checkpoint per block and
    step by one. The description is derived from the data rather than assumed,
    because the two layouts and the five depths give five different sets.
    """
    cks = sorted(cks)
    if len(cks) == 1:
        return str(cks[0])
    steps = set(np.diff(cks).tolist())
    if steps == {1}:
        return f'{cks[0]}–{cks[-1]}'
    if len(steps) == 1:
        step = steps.pop()
        return f'{cks[0]}, {cks[0] + step}, …, {cks[-1]}'
    return ', '.join(str(c) for c in cks)


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
    ax.set_ylabel('Mean UASL', fontsize=8)


def main():
    args = parse_args()
    npz  = np.load(args.curves)
    base = Path(args.curves).with_suffix('')
    out_path = (
        Path(args.out) if args.out
        else Path(str(base) + ('_logn.png' if args.log_x else '.png'))
    )

    all_checkpoints = npz['checkpoints'].tolist()

    # The original filtered the top panel on `ck % 2 == 1`, which is the
    # post-block parity of the 2+2L layout only; models stored as 1+L
    # (DeBERTa-v3-base, GPT-J) have post-block checkpoints at every index, so
    # take whatever the NPZ actually contains. The reference checkpoint is one
    # of them and is drawn in the top panel too, so that the panel is exactly
    # 'every post-block checkpoint' and its title can say so compactly.
    ref_ck   = args.ref_checkpoint
    main_cks = list(all_checkpoints)

    if ref_ck not in all_checkpoints:
        raise ValueError(
            f'Checkpoint {ref_ck} not found in NPZ (has {all_checkpoints}). '
            f'Re-run uuas_mean_curves_by_checkpoint.py with --checkpoints '
            f'including {ref_ck}, or pass a different --ref-checkpoint.'
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

    # The reference checkpoint keeps the colour it has in the top panel, so the
    # two panels can be read against each other. The original hard-coded a
    # colour interpolated between checkpoints 15 and 17, which only makes sense
    # for BERT-base's layout.
    color_16 = colors[ref_ck]

    # ── Figure: two stacked panels sharing x-axis ─────────────────────────────
    # The legend has one entry per checkpoint, so its height varies from two
    # rows (12 checkpoints) to five (GPT-J's 28). Reserve that space explicitly:
    # with a fixed figure height the legend box overlapped the x-axis label of
    # the deeper models.
    ncol = 6
    n_leg_rows = -(-len(main_cks) // ncol)
    leg_h = 0.155 * n_leg_rows + 0.30          # inches, incl. legend title
    panels_h = 5.4 - 0.20 * 5.4                # original panel area
    fig_h = panels_h + leg_h

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.5, fig_h), sharex=True,
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
    # Titles are derived from the data, not hard-coded: the checkpoint set and
    # the optimal checkpoint differ for every model (BERT-base 16, DeBERTa-v3
    # 9, ModernBERT 30, GPT-2 16, GPT-J 8), so a fixed '3 – 25' / '16' title
    # was wrong for four of the five.
    prefix = f'{args.model_label}: ' if args.model_label else ''
    ax_top.set_title(f'{prefix}post-block residual stream checkpoints '
                     f'({describe_checkpoints(main_cks)})', fontsize=8, pad=4)
    ax_top.tick_params(labelbottom=False)   # x labels hidden (shared axis)

    # — Bottom panel —
    xs16, means16, stds16, _ = get_curve(ref_ck)
    plot_curve(ax_bot, transform_x(xs16, args.log_x), means16, stds16,
               color_16, label=str(ref_ck))

    style_ax(ax_bot, args.log_x)
    xlabel = r'$\ln \delta$' if args.log_x else r'$\delta$ (linear distance, in words)'
    ax_bot.set_xlabel(xlabel, fontsize=8)
    ax_bot.set_title(f'{args.ref_desc} ({ref_ck})', fontsize=8, pad=4)

    # ── Legend below both panels, 6 columns ───────────────────────────────────
    fig.legend(
        handles=handles,
        labels=[str(ck) for ck in main_cks],
        title='Residual stream checkpoint',
        title_fontsize=7,
        fontsize=7,
        loc='lower center',
        ncol=ncol,
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=fig.transFigure,
        frameon=True,
        handlelength=1.4,
        columnspacing=0.8,
        handletextpad=0.4,
    )

    bottom = (leg_h + 0.30) / fig_h        # legend block plus the x-axis label
    fig.subplots_adjust(left=0.14, right=0.97, top=0.96, bottom=bottom,
                        hspace=0.12)

    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
