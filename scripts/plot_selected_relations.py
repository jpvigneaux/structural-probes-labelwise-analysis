#!/usr/bin/env python3
"""Plot per-relation ULAS against checkpoint index for a selected subset of relations.

Generalised from experiments/bert-base-prd/figures/plot_selected_relations.py,
which hard-coded BERT-base's results directory, output path and x-axis label.
Here the model is an argument, so the same figure can be produced for every
model in the comparison (BERT-base, DeBERTa-v3-base, ModernBERT-base, GPT-2,
GPT-J) and the appendix figures stay directly comparable to the main-text one.

Reads `training_uuas_by_relation.tsv` from each `layer-NN/` directory and takes,
per checkpoint, the epoch with the highest overall (micro-averaged) ULAS.

Usage:
    python plot_selected_relations.py --results-dir RESULTS \
        --out figs/selected_uuas_by_relation_gpt2.png --model-label "GPT-2-base"
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Same relation subset, colours and markers as the main-text figure, so the
# appendix panels can be read against it without a per-figure legend lookup.
# Two relation sets, one per main-text figure. Both were previously separate
# near-identical scripts; the only thing that differed was this list.
#
#   verb-args        the argument-structure figure: external vs internal
#                    arguments of the verb, coloured by that distinction
#   performance      the top / intermediate / low figure, with the intermediate
#                    band split into high, mid and low range
#
# Each set is a list of (group title, entries). The group title heads its block
# in the legend; `None` means the block is drawn without one, which is what the
# performance set wants -- its ordering is legible from the curves themselves,
# and for the shuffled model the BERT-derived band names would be misleading.
# Labels are bare relation names: the grouping is carried by the legend blocks.
RELATION_SETS = {
    'verb-args': [
        ('External arguments', [
            # red shades
            ('nsubj', 'nsubj', '#B00000', '-', 's'),
            ('csubj', 'csubj', '#E06060', '--', 'o'),
        ]),
        ('Internal arguments', [
            # blue shades
            ('dobj', 'dobj', '#08306B', '-', '^'),
            ('xcomp', 'xcomp', '#2171B5', '--', 'P'),
            ('iobj', 'iobj', '#4292C6', '-.', 'D'),
            ('pcomp', 'pcomp', '#9ECAE1', ':', 'h'),
        ]),
    ],
    'performance': [
        (None, [
            # Top performers -- blue;      marker: circle
            ('mwe',       'mwe',       '#08306B', '-',  'o'),
            ('auxpass',   'auxpass',   '#6BAED6', '--', 'o'),
            # Intermediate-high -- purple; marker: triangle-up
            ('cop',       'cop',       '#54278F', '-',  '^'),
            ('amod',      'amod',      '#B07FD4', '--', '^'),
            # Intermediate-mid -- orange;  marker: diamond
            ('cc',        'cc',        '#C14B00', '-',  'D'),
            ('poss',      'poss',      '#F5A048', '--', 'D'),
            # Intermediate-low -- teal;    marker: hexagon
            ('conj',      'conj',      '#005F6B', '-',  'h'),
            ('rcmod',     'rcmod',     '#40A8BD', '--', 'h'),
            # Low performers -- green;     marker: square
            ('advcl',     'advcl',     '#1A6B22', '-',  's'),
            ('parataxis', 'parataxis', '#5EC466', '--', 's'),
        ]),
    ],
}


def load(results_dir):
    records = []
    for layer_dir in sorted(Path(results_dir).glob('layer-*')):
        m = re.search(r'layer-(\d+)', layer_dir.name)
        if not m:
            continue
        tsv = layer_dir / 'training_uuas_by_relation.tsv'
        if not tsv.exists():
            continue
        df = pd.read_csv(tsv, sep='\t')
        epoch_uuas = (df.groupby('epoch')[['correct', 'total']].sum()
                      .assign(uuas=lambda x: x['correct'] / x['total']))
        best = df[df['epoch'] == epoch_uuas['uuas'].idxmax()][['relation', 'uuas']].copy()
        best['layer'] = int(m.group(1))
        records.append(best)
    if not records:
        raise SystemExit(f'no training_uuas_by_relation.tsv found under {results_dir}')
    return pd.concat(records, ignore_index=True)


def stack_legends(fig, ax, blocks, fontsize, title_fontsize):
    """Draw one legend per group, stacked down the right-hand side of the axes.

    Separate legends rather than one legend with heading rows: a legend title
    sits flush above its entries, whereas a heading faked with a blank handle
    would be indented by the handle column and would not read as a heading.
    Each block's height is measured after drawing, so the next one can be
    anchored just below it whatever the entry count or font metrics.
    """
    y = 1.0
    for title, handles, labels in blocks:
        leg = ax.legend(handles, labels, title=title, loc='upper left',
                        bbox_to_anchor=(1.02, y), fontsize=fontsize,
                        title_fontsize=title_fontsize, frameon=True,
                        framealpha=0.9, edgecolor='#cccccc',
                        borderpad=0.5, labelspacing=0.4, handlelength=2.0)
        if title is not None:
            leg.get_title().set_ha('left')
        ax.add_artist(leg)
        # add_artist clips to the axes rectangle, which would hide a legend
        # placed outside it.
        leg.set_clip_on(False)
        fig.canvas.draw()
        box = leg.get_window_extent().transformed(ax.transAxes.inverted())
        y = box.y0 - 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model-label', default='checkpoint')
    ap.add_argument('--relation-set', default='verb-args',
                    choices=sorted(RELATION_SETS),
                    help='which set of relations to draw (default: verb-args)')
    ap.add_argument('--tick-step', type=int, default=None,
                    help='x-tick spacing; default adapts to the checkpoint count')
    ap.add_argument('--figsize', default='4.6,3.1',
                    help='axes size in inches, "W,H"; the legend is added to '
                         'the right of this by the tight bounding box')
    args = ap.parse_args()

    data = load(args.results_dir)
    layers = sorted(data['layer'].unique())
    step = args.tick_step or max(1, round((layers[-1] - layers[0]) / 12))

    fig, ax = plt.subplots(figsize=tuple(float(v) for v in args.figsize.split(',')))
    blocks = []
    for title, entries in RELATION_SETS[args.relation_set]:
        handles, labels = [], []
        for rel, label, color, ls, marker in entries:
            sub = data[data['relation'] == rel].sort_values('layer')
            if sub.empty:
                print(f'WARNING: no data for {rel}')
                continue
            line, = ax.plot(sub['layer'], sub['uuas'], color=color, label=label,
                            linewidth=2.2, marker=marker, markersize=4.5,
                            linestyle=ls)
            handles.append(line)
            labels.append(label)
        if handles:
            blocks.append((title, handles, labels))

    ax.set_xlabel(f'{args.model_label} checkpoint index', fontsize=13)
    ax.set_ylabel('ULAS (optimal epoch)', fontsize=13)
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(layers[0], layers[-1] + 1, step))
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, color='#e0e0e0', linewidth=0.8)
    ax.set_facecolor('white')

    fig.tight_layout()
    stack_legends(fig, ax, blocks, fontsize=12, title_fontsize=11.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
