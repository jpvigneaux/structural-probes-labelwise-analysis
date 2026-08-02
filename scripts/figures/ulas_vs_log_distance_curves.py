#!/usr/bin/env python3
"""ULAS against log arc length, per relation, at one checkpoint.

The R^2 heat map (regression_uas_vs_log_distance.py) says how well the
log-linear decay model

    ULAS_n(r, k) = a + b ln(n + 1)

fits each relation, but not what the failures look like. This draws the
underlying data: for each relation, the observed ULAS at every arc length with
at least five gold edges, with the fitted line through it and its R^2. A
relation the heat map scores low is then visibly low for a reason -- a curve
that is flat, or humped, or noisy -- rather than merely having a small number
attached to it.

Drawn at a single checkpoint, because a relation's curve moves with depth and
overlaying all of them would defeat the purpose. Use the checkpoint whose ULAS
the rest of the analysis refers to: the paper uses BERT-base checkpoint 16,
which is where its peak ULAS, its regression and Table 1 all sit.

Relations are ordered by R^2, descending, so the panel order matches the heat
map's row order and the two figures can be read side by side.

Usage:
    python ulas_vs_log_distance_curves.py --curves curves.npz --checkpoint 16 \\
        --relations neg cop aux advcl det vmod ccomp cc \\
        --out ulas_vs_log_distance_ck16.png
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--curves', required=True, help='NPZ from uuas_mean_curves_by_checkpoint.py')
    ap.add_argument('--checkpoint', type=int, required=True)
    ap.add_argument('--relations', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ncols', type=int, default=4)
    ap.add_argument('--min-points', type=int, default=3)
    ap.add_argument('--figsize', default=None,
                    help='WIDTH,HEIGHT in inches. Give the size the figure will '
                         'be PRINTED at, so the font sizes below are the ones '
                         'the reader gets rather than being scaled down by LaTeX.')
    ap.add_argument('--title-font', type=float, default=8.5)
    ap.add_argument('--label-font', type=float, default=8.5)
    ap.add_argument('--tick-font', type=float, default=7.0)
    ap.add_argument('--marker-size', type=float, default=13.0)
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--model-label', default='')
    args = ap.parse_args()

    npz = np.load(args.curves, allow_pickle=False)
    checkpoints = npz['checkpoints'].tolist()
    relations = npz['rel_names'].astype(str).tolist()

    if args.checkpoint not in checkpoints:
        raise SystemExit(
            f'checkpoint {args.checkpoint} is not in {args.curves} '
            f'(has {checkpoints}). Rebuild the curves with it included.')
    ci = checkpoints.index(args.checkpoint)

    missing = [r for r in args.relations if r not in relations]
    if missing:
        raise SystemExit(f'not in the curves file: {missing}')

    # Fit first, so the panels can be ordered by goodness of fit like the heat map.
    panels = []
    for rel in args.relations:
        ri = relations.index(rel)
        ns = npz[f'rel_ns_{ci}_{ri}'].astype(float)
        uas = npz[f'rel_uas_{ci}_{ri}'].astype(float)
        if len(ns) < args.min_points:
            panels.append((rel, ns, uas, None))
            continue
        panels.append((rel, ns, uas, stats.linregress(np.log(ns + 1), uas)))
    panels.sort(key=lambda t: (t[3].rvalue ** 2) if t[3] else -1, reverse=True)

    ncols = min(args.ncols, len(panels))
    nrows = -(-len(panels) // ncols)
    if args.figsize:
        w, h = (float(v) for v in args.figsize.split(','))
    else:
        w, h = 1.85 * ncols + 0.5, 1.75 * nrows + 0.55
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h),
                             sharex=True, sharey=True, facecolor='white')
    axes = np.atleast_1d(axes).ravel()

    for ax, (rel, ns, uas, fit) in zip(axes, panels):
        x = np.log(ns + 1)
        ax.scatter(x, uas, s=args.marker_size, color='#08306B', zorder=3, alpha=0.85)
        if fit is not None:
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, fit.slope * xs + fit.intercept, color='#C14B00',
                    lw=1.4, zorder=2)
            r2 = fit.rvalue ** 2
            # Colour the annotation by fit quality, on the heat map's scale, so
            # a reader moving between the two figures sees the same signal.
            col = '#1A6B22' if r2 >= 0.75 else ('#C14B00' if r2 >= 0.4 else '#B00000')
            ax.set_title(f'{rel}  $R^2$={r2:.2f}', fontsize=args.title_font,
                         color=col, pad=2)
        else:
            ax.set_title(f'{rel}  (too few)', fontsize=args.title_font,
                         color='#777777', pad=2)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=args.tick_font)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    # One label per edge rather than per panel.
    for ax in axes[len(panels) - ncols:len(panels)]:
        ax.set_xlabel(r'$\ln(n+1)$', fontsize=args.label_font)
    for ax in axes[::ncols]:
        ax.set_ylabel('ULAS', fontsize=args.label_font)

    if args.model_label:
        fig.suptitle(f'{args.model_label}, checkpoint {args.checkpoint}',
                     fontsize=args.label_font + 1.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98 if args.model_label else 1.0))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f'Saved → {args.out}')
    for rel, ns, uas, fit in panels:
        r2 = f'{fit.rvalue ** 2:.3f}' if fit else '  n/a'
        print(f'  {rel:<10s} R^2 = {r2}   {len(ns):2d} arc lengths, '
              f'ULAS {uas.min():.3f}-{uas.max():.3f}')


if __name__ == '__main__':
    main()
