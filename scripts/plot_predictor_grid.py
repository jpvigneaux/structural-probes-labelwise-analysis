#!/usr/bin/env python3
"""Grid of ULAS against each regression predictor, one row per model.

Rows are models, columns are the three predictors of the WLS regression:
mean(log arc length), sd(log arc length) and head similarity-corrected entropy.

The predictors are computed once on the Penn Treebank, so they are identical
for every model: each column therefore shares an x-axis exactly, and the rows
differ only in the ULAS values on the y-axis. That makes the grid a direct
visual comparison -- the same 42 points move vertically from row to row.

Point area is proportional to the number of dev-set edges the relation's ULAS
was measured from, i.e. to its weight in the regression, so the visually
dominant points are the statistically reliable ones. The line in each panel is
the weighted least-squares fit of the marginal relationship; the full model
controls for the other two predictors, so these slopes are marginal, not
partial, and are shown to convey direction and spread.

One cell of the grid can also be drawn on its own, with --panel, for use in the
main text at a size where the relation labels fit. That is a different rendering
of the same coordinates, not a different computation: the residualisation is
per-model, so a panel taken out of the grid is unchanged except for the axis
limits, which are shared down a column in the grid and set from the panel's own
data when it stands alone.

Usage:
    python plot_predictor_grid.py --spec "BERT-base:RESULTS:16" \\
        --spec "GPT-2-base:RESULTS2:16" --sim sim.tsv --len lengths.tsv --out grid.png

    python plot_predictor_grid.py --spec "BERT-base:RESULTS:16" \\
        --sim sim.tsv --len lengths.tsv --panel head_sim_entropy \\
        --annotate 12 --out head_sim_entropy_bertbase.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
POINT = '#2a78d6'
FIT = '#eb6834'

# (column key, axis label)
PREDICTORS = [
    ('mean_log_length', r'mean $\log$ arc length'),
    ('sd_log_length', r'sd $\log$ arc length'),
    ('head_sim_entropy', 'head sim-entropy (bits)'),
]


def weighted_fit(x, y, w):
    W = w / w.sum()
    xm, ym = np.sum(W * x), np.sum(W * y)
    sxx = np.sum(W * (x - xm) ** 2)
    sxy = np.sum(W * (x - xm) * (y - ym))
    syy = np.sum(W * (y - ym) ** 2)
    slope = sxy / sxx
    return slope, ym - slope * xm, sxy / np.sqrt(sxx * syy)


def wls_beta(y, X, w):
    """Weighted least-squares coefficients of y on X, intercept first."""
    A = np.column_stack([np.ones(len(y)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    return beta


def wls_resid(y, X, w, y_apply=None):
    """Residuals of a weighted least-squares fit of y on X (intercept added).

    With `y_apply`, the fit is still estimated from `y` but the residual is
    taken of `y_apply`. That is how the held-out panels are built: the
    adjustment for the other predictors is estimated on the dev split, where
    the regression is fitted, and then applied to the test-split ULAS.
    """
    A = np.column_stack([np.ones(len(y)), X])
    beta = wls_beta(y, X, w)
    return (y if y_apply is None else y_apply) - A @ beta


def partial_axes(df, key, keys, w, response='uuas'):
    """Added-variable (partial regression) coordinates for predictor `key`.

    Residualise both ULAS and `key` on the *other* predictors. The weighted
    slope through the resulting cloud equals this predictor's coefficient in
    the full multiple regression, so the panel agrees with the reported table.
    That equality is the Frisch-Waugh-Lovell theorem (Frisch and Waugh 1933;
    Lovell 1963); for the plot built on it see Belsley, Kuh and Welsch (1980) or
    Cook and Weisberg (1982). The paper states the construction in its appendix
    "How to read the predictor figures".

    This matters here: sd(log n) correlates +0.56 with mean(log n), which is
    itself strongly negative for ULAS, so a raw scatter of ULAS against
    sd(log n) slopes downward even though the partial effect is positive.

    With `response='test_uuas'` the vertical coordinate is the held-out ULAS,
    adjusted by the dev-estimated contribution of the other predictors and
    plotted against an x-axis that is unchanged, since the predictors are
    corpus properties and do not depend on the split. Nothing about the model
    is re-estimated on the test edges, so the drawn line remains the dev fit
    and the points are a genuine out-of-sample comparison against it.
    """
    others = [k for k in keys if k != key]
    X = df[others].to_numpy(dtype=float)
    y_dev = df['uuas'].to_numpy(dtype=float)
    y_apply = None if response == 'uuas' else df[response].to_numpy(dtype=float)
    ry = wls_resid(y_dev, X, w, y_apply=y_apply)
    rx = wls_resid(df[key].to_numpy(dtype=float), X, w)
    return rx, ry


def read_by_relation(path):
    u = pd.read_csv(path, sep='\t')
    if 'total' not in u.columns and {'correct', 'uuas'}.issubset(u.columns):
        u['total'] = u['correct'] / u['uuas']
    return u.set_index('relation')


def load_model(uuas_path, sim, length, test_path=None):
    u = read_by_relation(uuas_path)
    c = u.index.intersection(sim.index).intersection(length.index)
    cols = {
        'uuas': u.loc[c, 'uuas'],
        'total': u.loc[c, 'total'],
        'mean_log_length': length.loc[c, 'mean_log_length'],
        'sd_log_length': length.loc[c, 'stdev_log_length'],
        'head_sim_entropy': sim.loc[c, 'head_sim_entropy_bits'],
    }
    if test_path is not None:
        t = read_by_relation(test_path)
        # Relations absent from the test split are dropped by the dropna below.
        cols['test_uuas'] = t['uuas'].reindex(c)
        cols['test_total'] = t['total'].reindex(c)
    return pd.DataFrame(cols).dropna()


def response_column(args):
    return 'uuas' if args.ulas == 'dev' else 'test_uuas'


def area_column(args):
    """Which edge count sets a point's area.

    The area says how precisely that point's ULAS is measured, so it follows
    the split the ULAS is measured on -- which for the dev panels is also the
    relation's weight in the regression.
    """
    return 'total' if args.ulas == 'dev' else 'test_total'


def ylabel_for(args):
    stem = 'test ULAS' if args.ulas == 'test' else 'ULAS'
    return stem if args.marginal else f'{stem}, residual'


def draw_panel(label, df, key, xlabel, args):
    """One predictor, one model, as a standalone figure.

    Same coordinates as the corresponding cell of the grid; only the axis limits
    differ, being set from this panel's own data rather than shared down a column.
    """
    keys = [k for k, _ in PREDICTORS]
    resp = response_column(args)
    w = df['total'].to_numpy(dtype=float)              # regression weights: dev
    area_w = df[area_column(args)].to_numpy(dtype=float)
    if args.marginal:
        x = df[key].to_numpy(dtype=float)
        y, y_dev = df[resp].to_numpy(dtype=float), df['uuas'].to_numpy(dtype=float)
        x_dev = x
    else:
        x, y = partial_axes(df, key, keys, w, response=resp)
        x_dev, y_dev = partial_axes(df, key, keys, w)
    # The line is always the dev fit, so that a held-out panel is compared
    # against the published model rather than against one refitted on itself.
    slope, intercept, _ = weighted_fit(x_dev, y_dev, w)
    got, _, corr = weighted_fit(x, y, area_w)

    fw, fh = (float(v) for v in args.figsize.split(','))
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if not args.marginal:
        ax.axhline(0, color=AXIS, linewidth=0.7, zorder=1)
        ax.axvline(0, color=AXIS, linewidth=0.7, zorder=1)

    sizes = 6 + 150 * (area_w / area_w.max())
    ax.scatter(x, y, s=sizes, alpha=0.5, color=POINT, edgecolor='white',
               linewidth=0.5, zorder=3)

    padx = 0.06 * (x.max() - x.min())
    pady = 0.08 * (y.max() - y.min())
    xlim = (x.min() - padx, x.max() + padx)
    xs = np.linspace(*xlim, 50)
    ax.plot(xs, intercept + slope * xs, color=FIT, linewidth=1.6, zorder=4)
    ax.set_xlim(*xlim)
    ax.set_ylim(y.min() - pady, y.max() + pady)

    if args.annotate:
        # The heaviest relations are the ones that determine the slope, so those
        # are the ones worth naming; a label goes clear of its own marker, whose
        # radius in points is sqrt(area)/2.
        order = np.argsort(area_w)[::-1][:args.annotate]
        rels = df.index.to_numpy()
        flip = xlim[0] + 0.78 * (xlim[1] - xlim[0])   # label leftwards near the right edge
        for i in order:
            off = np.sqrt(sizes[i]) / 2 + 1.5
            left = x[i] > flip
            ax.annotate(rels[i], (x[i], y[i]), textcoords='offset points',
                        xytext=(-off if left else off, off * 0.5),
                        ha='right' if left else 'left',
                        fontsize=args.annotate_font, color=INK_2, zorder=5)

    ax.text(0.97, 0.94, rf'$\beta={slope:+.3f}$', transform=ax.transAxes,
            ha='right', va='top', fontsize=8, color=INK)
    ax.set_xlabel(xlabel if args.marginal else f'{xlabel}, residual',
                  fontsize=8.5, color=INK)
    ax.set_ylabel(ylabel_for(args), fontsize=8.5, color=INK)
    ax.tick_params(labelsize=7, length=0, colors=INK_2)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(AXIS)

    fig.tight_layout(pad=0.4)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    kind = 'marginal' if args.marginal else 'partial'
    print(f'Saved {args.out}  ({label}, {key}, {kind}, {args.ulas} ULAS, n = {len(df)})')
    print(f'  dev-fitted slope = {slope:+.4f}   weighted r = {corr:+.4f}')
    if args.ulas == 'test':
        print(f'  slope through the plotted test points = {got:+.4f} '
              f'({abs(got - slope) / abs(slope):.1%} from the dev fit, not refitted)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', action='append', required=True,
                    help='LABEL:RESULTS_DIR:CHECKPOINT (repeatable, one per row)')
    ap.add_argument('--sim', required=True)
    ap.add_argument('--len', dest='length', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=float, default=6.9)
    ap.add_argument('--row-height', type=float, default=1.72)
    ap.add_argument('--ulas', choices=('dev', 'test'), default='dev',
                    help="which split's ULAS goes on the vertical axis. With "
                         "'test' the regression is still fitted on dev -- the "
                         "line, and the adjustment for the other predictors, "
                         "come from the dev fit and only the points are "
                         "held-out, so the panel shows prediction rather than "
                         "fit. Relations absent from the test split are dropped.")
    ap.add_argument('--panel', default=None, metavar='PREDICTOR',
                    help='draw only this predictor, for one model, as a standalone '
                         f'figure. One of: {", ".join(k for k, _ in PREDICTORS)}.')
    ap.add_argument('--panel-model', default=None, metavar='LABEL',
                    help='which --spec the panel comes from (default: the first)')
    ap.add_argument('--figsize', default='3.15,2.65', metavar='W,H',
                    help='size of the standalone panel, in inches. Give the size '
                         'it will be printed at, so the font sizes are the ones '
                         'the reader gets.')
    ap.add_argument('--annotate', type=int, default=0, metavar='N',
                    help='label the N heaviest relations in the standalone panel')
    ap.add_argument('--annotate-font', type=float, default=5.6)
    ap.add_argument('--marginal', action='store_true',
                    help='plot raw ULAS against each predictor instead of the '
                         'added-variable (partial) view. Note that the marginal '
                         'slope for sd(log n) has the opposite sign to its '
                         'regression coefficient, because it is confounded with '
                         'mean(log n).')
    args = ap.parse_args()

    sim = pd.read_csv(args.sim, sep='\t').set_index('deprel')
    length = pd.read_csv(args.length, sep='\t').set_index('deprel')

    rows = []
    for spec in args.spec:
        label, results, ck = spec.rsplit(':', 2)
        layer = Path(results) / f'layer-{int(ck):02d}'
        test = layer / 'test.uuas_by_relation' if args.ulas == 'test' else None
        rows.append((label, load_model(layer / 'dev.uuas_by_relation', sim,
                                       length, test_path=test)))

    if args.panel:
        labels = dict(PREDICTORS)
        if args.panel not in labels:
            raise SystemExit(f'--panel must be one of {sorted(labels)}')
        if args.panel_model:
            chosen = [r for r in rows if r[0] == args.panel_model]
            if not chosen:
                raise SystemExit(f'no --spec labelled {args.panel_model!r}; '
                                 f'have {[r[0] for r in rows]}')
        else:
            chosen = rows[:1]
        draw_panel(chosen[0][0], chosen[0][1], args.panel, labels[args.panel], args)
        return

    n = len(rows)
    keys = [k for k, _ in PREDICTORS]
    fig, axes = plt.subplots(n, 3, figsize=(args.width, args.row_height * n),
                             sharey=args.marginal, facecolor=SURFACE)
    if n == 1:
        axes = axes[None, :]

    # Precompute coordinates so the axis limits can be shared per column. Each
    # entry is (x, y, area weights, dev-fitted line), the line being kept
    # separate from the points so a held-out panel is drawn against the fit
    # rather than against a slope refitted on the plotted data.
    resp = response_column(args)
    coords = {}
    for r, (label, df) in enumerate(rows):
        w = df['total'].to_numpy(dtype=float)
        area_w = df[area_column(args)].to_numpy(dtype=float)
        for key in keys:
            if args.marginal:
                x = df[key].to_numpy(dtype=float)
                y, y_dev = df[resp].to_numpy(dtype=float), df['uuas'].to_numpy(dtype=float)
                x_dev = x
            else:
                x, y = partial_axes(df, key, keys, w, response=resp)
                x_dev, y_dev = partial_axes(df, key, keys, w)
            coords[(r, key)] = (x, y, area_w, weighted_fit(x_dev, y_dev, w))

    xlims, ylims = {}, {}
    for key in keys:
        allx = np.concatenate([coords[(r, key)][0] for r in range(n)])
        pad = 0.06 * (allx.max() - allx.min())
        xlims[key] = (allx.min() - pad, allx.max() + pad)
    if args.marginal:
        ylims = {key: (-0.03, 1.05) for key in keys}
    else:
        ally = np.concatenate([coords[(r, k)][1] for r in range(n) for k in keys])
        pad = 0.08 * (ally.max() - ally.min())
        ylims = {key: (ally.min() - pad, ally.max() + pad) for key in keys}

    for r, (label, df) in enumerate(rows):
        for c, (key, xlabel) in enumerate(PREDICTORS):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            x, y, w, (slope, intercept, _) = coords[(r, key)]

            ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
            if not args.marginal:
                ax.axhline(0, color=AXIS, linewidth=0.7, zorder=1)
                ax.axvline(0, color=AXIS, linewidth=0.7, zorder=1)
            ax.scatter(x, y, s=6 + 150 * (w / w.max()), alpha=0.5, color=POINT,
                       edgecolor='white', linewidth=0.5, zorder=3)
            xs = np.linspace(*xlims[key], 50)
            ax.plot(xs, intercept + slope * xs, color=FIT, linewidth=1.6, zorder=4)
            ax.text(0.96, 0.93, rf'$\beta={slope:+.3f}$', transform=ax.transAxes,
                    ha='right', va='top', fontsize=7.5, color=INK)

            ax.set_xlim(*xlims[key])
            ax.set_ylim(*ylims[key])
            ax.tick_params(labelsize=7, length=0, colors=INK_2)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            for s in ('left', 'bottom'):
                ax.spines[s].set_color(AXIS)

            if r == 0:
                ax.set_title(xlabel, fontsize=8.5, color=INK, pad=6)
            if r < n - 1:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(label, fontsize=8.5, color=INK)

    if args.ulas == 'test':
        fig.supylabel(ylabel_for(args), fontsize=8.5, color=INK_2, x=0.005)
    fig.tight_layout(h_pad=0.6, w_pad=0.7)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight', facecolor=SURFACE)
    print(f'Saved {args.out}  ({n} rows x 3 columns)')

    kind = 'marginal correlations' if args.marginal else 'partial slopes (= regression coefficients)'
    print(f'\nweighted {kind}, from the dev fit (the line drawn in each panel):')
    print(f'{"model":18s} ' + ' '.join(f'{k:>18s}' for k in keys))
    for r, (label, df) in enumerate(rows):
        vals = []
        for key in keys:
            _, _, _, (slope, _, corr) = coords[(r, key)]
            vals.append(corr if args.marginal else slope)
        print(f'{label:18s} ' + ' '.join(f'{v:>18.4f}' for v in vals))

    if args.ulas == 'test':
        print('\nsame slopes, refitted on the plotted test points '
              '(not drawn; for comparison only):')
        for r, (label, df) in enumerate(rows):
            vals = []
            for key in keys:
                x, y, w, _ = coords[(r, key)]
                slope, _, corr = weighted_fit(x, y, w)
                vals.append(corr if args.marginal else slope)
            print(f'{label:18s} ' + ' '.join(f'{v:>18.4f}' for v in vals))


if __name__ == '__main__':
    main()
