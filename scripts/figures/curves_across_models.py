#!/usr/bin/env python3
"""Mean UASL against linear distance, one curve per model at its own optimum.

The main-text figure used to show BERT-base alone, in two panels: every
post-block checkpoint above, the best one below. This draws only the second of
those, and puts every model on it, so the decay is shown to be a property of
the representations rather than of one encoder.

Each model contributes the checkpoint reported as its optimum in the
cross-model table, which for BERT-base, GPT-2 and ModernBERT is a
post-attention checkpoint and so is not among the post-block ones the
per-model appendix figures draw.

Colours are the Okabe-Ito qualitative set in an order checked for
colour-vision separation; line style and marker repeat the distinction, so the
curves stay separable in greyscale and in print.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# (key, label, colour, linestyle, marker). Colour is fixed per model and never
# cycled: the same model keeps its colour if the set is ever cut down.
MODELS = [
    ('bertbase',   'BERT-base',       '#0072B2', '-',   'o'),
    ('deberta',    'DeBERTa-v3-base', '#D55E00', '--',  's'),
    ('modernbert', 'ModernBERT-base', '#009E73', '-.',  '^'),
    ('gpt2',       'GPT-2-base',      '#E69F00', ':',   'D'),
    ('gptj',       'GPT-J-6B',        '#CC79A7', (0, (3, 1, 1, 1)), 'v'),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--npz-dir', required=True,
                   help='directory holding opt_<model>.npz, one per model')
    p.add_argument('--out', required=True)
    p.add_argument('--band', action='store_true',
                   help='shade one standard deviation across relations; off by '
                        'default because five overlapping bands are unreadable')
    p.add_argument('--figsize', default='3.15,2.5',
                   help='inches, W,H; the default is one ACL column')
    return p.parse_args()


def main():
    args = parse_args()
    fig, ax = plt.subplots(figsize=tuple(float(v) for v in args.figsize.split(',')))

    for key, label, colour, ls, marker in MODELS:
        z = np.load(Path(args.npz_dir) / f'opt_{key}.npz', allow_pickle=True)
        ck = int(z['checkpoints'][0])
        xs, means, stds = z['xs_0'], z['means_0'], z['stds_0']
        if args.band:
            ax.fill_between(xs, np.maximum(means - stds, 0.0), means + stds,
                            alpha=0.10, color=colour, linewidth=0)
        ax.plot(xs, means, linestyle=ls, marker=marker, color=colour,
                linewidth=1.2, markersize=2.8, label=f'{label} ({ck})')

    ax.set_ylim(0, 1.02)
    ax.set_xlim(left=0.4)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', linewidth=0.4, alpha=0.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylabel('Mean UASL', fontsize=8)
    ax.set_xlabel(r'$\delta$ (linear distance, in words)', fontsize=8)
    # identity is never colour alone: every series is named in the legend, and
    # the parenthesised number is the checkpoint it was read at
    ax.legend(fontsize=6.2, frameon=True, framealpha=0.9, edgecolor='#cccccc',
              borderpad=0.4, labelspacing=0.3, handlelength=2.4, loc='upper right')

    fig.tight_layout(pad=0.3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
