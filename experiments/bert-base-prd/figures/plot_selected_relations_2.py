"""Plot per-relation ULAS vs. BERT-base layer: five performance tiers."""

import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = Path('{REPO_ROOT}/experiments/bert-base-prd/results-hface')
OUT = Path('{REPO_ROOT}/experiments/bert-base-prd/figures/selected_uuas_by_relation_2.png')

# ---------------------------------------------------------------------------
# Load data
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
# Relations: (key, label, color, linestyle)
# Five tiers, each with a distinct hue; two shades + two line styles per tier.
# ---------------------------------------------------------------------------
RELATIONS = [
    # Top performers — blue;         marker: circle
    ('mwe',       'mwe (Top)',          '#08306B', '-',  'o'),
    ('auxpass',   'auxpass (Top)',       '#6BAED6', '--', 'o'),
    # Intermediate-high — purple;    marker: triangle-up
    ('cop',       'cop (Inter-high)',    '#54278F', '-',  '^'),
    ('amod',      'amod (Inter-high)',   '#B07FD4', '--', '^'),
    # Intermediate-mid — orange;     marker: diamond
    ('cc',        'cc (Inter-mid)',      '#C14B00', '-',  'D'),
    ('poss',      'poss (Inter-mid)',    '#F5A048', '--', 'D'),
    # Intermediate-low — teal;       marker: hexagon
    ('conj',      'conj (Inter-low)',    '#005F6B', '-',  'h'),
    ('rcmod',     'rcmod (Inter-low)',   '#40A8BD', '--', 'h'),
    # Low performers — green;        marker: square
    ('advcl',     'advcl (Low)',         '#1A6B22', '-',  's'),
    ('parataxis', 'parataxis (Low)',     '#5EC466', '--', 's'),
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
all_layers = sorted(data['layer'].unique())
ax.set_xlim(all_layers[0] - 0.5, all_layers[-1] + 0.5)
ax.set_ylim(0, 1.02)
ax.set_xticks(range(all_layers[0], all_layers[-1] + 1, 2))
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, color='#e0e0e0', linewidth=0.8)
ax.set_facecolor('white')

ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.16),
    ncol=2,
    fontsize=10.5,
    title="Legend: Dependency relation (Performance)",
    title_fontsize=9.5,
    frameon=True,
    framealpha=0.9,
    edgecolor='#cccccc',
)

fig.tight_layout()
fig.subplots_adjust(bottom=0.30)
fig.savefig(OUT, dpi=300, bbox_inches='tight')
print(f'Saved to {OUT}')
