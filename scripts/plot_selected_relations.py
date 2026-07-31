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
RELATION_SETS = {
    'verb-args': [
        # External arguments -- red shades
        ('nsubj', 'nsubj (Ext. Arg)', '#B00000', '-', 's'),
        ('csubj', 'csubj (Ext. Arg)', '#E06060', '--', 'o'),
        # Internal arguments -- blue shades
        ('dobj', 'dobj (Int. Arg)', '#08306B', '-', '^'),
        ('xcomp', 'xcomp (Int. Arg)', '#2171B5', '--', 'P'),
        ('iobj', 'iobj (Int. Arg)', '#4292C6', '-.', 'D'),
        ('pcomp', 'pcomp (Int. Arg)', '#9ECAE1', ':', 'h'),
    ],
    'performance': [
        # Top performers -- blue;      marker: circle
        ('mwe',       'mwe (Top)',          '#08306B', '-',  'o'),
        ('auxpass',   'auxpass (Top)',      '#6BAED6', '--', 'o'),
        # Intermediate-high -- purple; marker: triangle-up
        ('cop',       'cop (Inter-high)',   '#54278F', '-',  '^'),
        ('amod',      'amod (Inter-high)',  '#B07FD4', '--', '^'),
        # Intermediate-mid -- orange;  marker: diamond
        ('cc',        'cc (Inter-mid)',     '#C14B00', '-',  'D'),
        ('poss',      'poss (Inter-mid)',   '#F5A048', '--', 'D'),
        # Intermediate-low -- teal;    marker: hexagon
        ('conj',      'conj (Inter-low)',   '#005F6B', '-',  'h'),
        ('rcmod',     'rcmod (Inter-low)',  '#40A8BD', '--', 'h'),
        # Low performers -- green;     marker: square
        ('advcl',     'advcl (Low)',        '#1A6B22', '-',  's'),
        ('parataxis', 'parataxis (Low)',    '#5EC466', '--', 's'),
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
    args = ap.parse_args()

    data = load(args.results_dir)
    layers = sorted(data['layer'].unique())
    step = args.tick_step or max(1, round((layers[-1] - layers[0]) / 12))

    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for rel, label, color, ls, marker in RELATION_SETS[args.relation_set]:
        sub = data[data['relation'] == rel].sort_values('layer')
        if sub.empty:
            print(f'WARNING: no data for {rel}')
            continue
        ax.plot(sub['layer'], sub['uuas'], color=color, label=label,
                linewidth=2.2, marker=marker, markersize=4.5, linestyle=ls)

    ax.set_xlabel(f'{args.model_label} checkpoint index', fontsize=12)
    ax.set_ylabel('ULAS (optimal epoch)', fontsize=12)
    ax.set_xlim(layers[0] - 0.5, layers[-1] + 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(layers[0], layers[-1] + 1, step))
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, color='#e0e0e0', linewidth=0.8)
    ax.set_facecolor('white')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2,
              fontsize=10.5, title='Legend: Relation', title_fontsize=9.5,
              frameon=True, framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
