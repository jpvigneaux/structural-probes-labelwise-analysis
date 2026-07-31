"""Plot per-relation ULAS vs. BERT-base layer for a selected subset of relations."""

import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path('{REPO_ROOT}/experiments/bert-base-prd/results-hface')
OUT = Path('{REPO_ROOT}/experiments/bert-base-prd/figures/selected_uuas_by_relation.png')

# ---------------------------------------------------------------------------
# Load data (mirrors plot_optimal_uuas_by_relation_and_layer.py logic)
# ---------------------------------------------------------------------------
records = []
for layer_dir in sorted(RESULTS_DIR.glob('layer-*')):
    m = re.search(r'layer-(\d+)', layer_dir.name)
    if not m:
        continue
    layer = int(m.group(1))
    tsv = layer_dir / 'training_uuas_by_relation.tsv'
    if not tsv.exists():
        continue
    df = pd.read_csv(tsv, sep='\t')
    # Best epoch = max reconstructed overall UUAS
    epoch_uuas = (
        df.groupby('epoch')[['correct', 'total']]
        .sum()
        .assign(uuas=lambda x: x['correct'] / x['total'])
    )
    best_epoch = epoch_uuas['uuas'].idxmax()
    best = df[df['epoch'] == best_epoch][['relation', 'uuas']].copy()
    best['layer'] = layer
    records.append(best)

data = pd.concat(records, ignore_index=True)

# ---------------------------------------------------------------------------
# Relations to plot
# ---------------------------------------------------------------------------
RELATIONS = [
    # External arguments — red shades; distinct markers
    ('nsubj', 'nsubj (Ext. Arg)', '#B00000', '-',  's'),
    ('csubj', 'csubj (Ext. Arg)', '#E06060', '--', 'o'),
    # Internal arguments — blue shades; distinct markers
    ('dobj',  'dobj (Int. Arg)',  '#08306B', '-',  '^'),
    ('xcomp', 'xcomp (Int. Arg)', '#2171B5', '--', 'P'),
    ('iobj',  'iobj (Int. Arg)',  '#4292C6', '-.',  'D'),
    ('pcomp', 'pcomp (Int. Arg)', '#9ECAE1', ':',  'h'),
]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 4.6))

for rel, label, color, ls, marker in RELATIONS:
    sub = data[data['relation'] == rel].sort_values('layer')
    if sub.empty:
        print(f'WARNING: no data for {rel}')
        continue
    ax.plot(sub['layer'], sub['uuas'],
            color=color, label=label, linewidth=2.2,
            marker=marker, markersize=4.5, linestyle=ls)

ax.set_xlabel('BERT-base checkpoint index', fontsize=12)
ax.set_ylabel('ULAS (optimal epoch)', fontsize=12)
ax.set_xlim(-0.5, 25.5)
ax.set_ylim(0, 1.02)
all_layers = sorted(data['layer'].unique())
ax.set_xticks(range(all_layers[0], all_layers[-1] + 1, 2))
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, color='#e0e0e0', linewidth=0.8)
ax.set_facecolor('white')

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
    fontsize=10.5,
    title='Legend: Relation',
    title_fontsize=9.5,
    frameon=True,
    framealpha=0.9,
    edgecolor='#cccccc',
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.30)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT}')
