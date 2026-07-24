#!/usr/bin/env python3
"""
Compute UUAS-by-relation-by-linear-distance for all 26 BERT layer checkpoints
and write a single self-contained interactive HTML.

The figure has two dropdowns:
  • Checkpoint  – selects layer 0 … 25
  • Relation    – selects a PTB dependency-relation label

Always uses the TEST split.

Typical invocation (Quest CPU node):
    python uuas_by_relation_distance_all_layers.py \
        [--trained-probes-dir results-hface]       \
        [--out figures/uuas_by_relation_distance_all_layers.html]
"""
import argparse
import json
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
import plotly.graph_objects as go

# ── CoNLL-X parser ─────────────────────────────────────────────────────────────

_FIELDS = [
    'index', 'sentence', 'lemma_sentence', 'upos_sentence', 'xpos_sentence',
    'morph', 'head_indices', 'governance_relations', 'secondary_relations',
    'extra_info', 'embeddings',
]
Observation = namedtuple('Observation', _FIELDS)


def load_conll(path):
    observations, buf = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if not line:
                if buf:
                    cols = list(zip(*[tok.split('\t') for tok in buf]))
                    observations.append(Observation(*cols, [None] * len(cols[0])))
                    buf = []
            else:
                buf.append(line)
    if buf:
        cols = list(zip(*[tok.split('\t') for tok in buf]))
        observations.append(Observation(*cols, [None] * len(cols[0])))
    return observations


# ── Structural probe ───────────────────────────────────────────────────────────

class TwoWordPSDProbe(nn.Module):
    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.proj = nn.Parameter(torch.zeros(hidden_dim, rank))

    def forward(self, embeddings):
        t = embeddings @ self.proj
        diff = t.unsqueeze(0) - t.unsqueeze(1)
        return diff.pow(2).sum(-1)


# ── Prim's MST (matches reporter.py) ──────────────────────────────────────────

PUNCT_POS = {"''", ",", ".", ":", "``", "-LRB-", "-RRB-"}


class _UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            i = self.parents[i]
        return i

    def union(self, i, j):
        self.parents[self.find(i)] = j


def mst_edges(dist_matrix, poses):
    n = len(poses)
    pairs = {
        (i, j): dist_matrix[i, j].item()
        for i in range(n)
        for j in range(n)
        if poses[i] not in PUNCT_POS and poses[j] not in PUNCT_POS
    }
    uf = _UnionFind(n)
    edges = []
    for (i, j), _ in sorted(pairs.items(), key=lambda x: x[1]):
        if uf.find(i) != uf.find(j):
            uf.union(i, j)
            edges.append((i, j))
    return edges


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trained-probes-dir',
                   default=str(Path(__file__).parent / 'results-hface'),
                   help='Directory containing layer-00 … layer-25 subdirs')
    p.add_argument('--out', default=None,
                   help='Output HTML path '
                        '(default: <script-dir>/figures/uuas_by_relation_distance_all_layers.html)')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tp_dir = Path(args.trained_probes_dir)
    script_dir = Path(__file__).parent
    out_path = (
        Path(args.out) if args.out
        else script_dir / 'figures' / 'uuas_by_relation_distance_all_layers.html'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    split = 'test'
    n_layers = 26

    # ── Common paths from layer-00 config ────────────────────────────────────
    layer0_dir = tp_dir / 'layer-00'
    config_files = list(layer0_dir.glob('*.yaml'))
    if not config_files:
        sys.exit(f'No YAML config found in {layer0_dir}')
    with open(config_files[0]) as f:
        cfg0 = yaml.safe_load(f)

    corpus_root = Path(cfg0['dataset']['corpus']['root'])
    emb_root    = Path(cfg0['dataset']['embeddings']['root'])
    align_root  = Path(cfg0['model']['alignment_root'])
    conll_path  = corpus_root / cfg0['dataset']['corpus'][f'{split}_path']
    emb_path    = emb_root    / cfg0['dataset']['embeddings'][f'{split}_path']
    align_path  = align_root  / cfg0['model'][f'alignment_{split}_path']

    print(f'split        : {split}')
    print(f'conll        : {conll_path}')
    print(f'embeddings   : {emb_path}')
    print(f'alignments   : {align_path}')

    # ── Load CoNLL data ───────────────────────────────────────────────────────
    print('\nLoading CoNLL data …')
    observations = load_conll(conll_path)
    n_sentences  = len(observations)
    print(f'  {n_sentences} sentences')

    # ── Load alignments (shared across layers) ────────────────────────────────
    print('Loading alignments …')
    alignments = []
    with h5py.File(align_path, 'r') as af:
        for idx in range(n_sentences):
            alignments.append(
                torch.tensor(np.array(af[str(idx)]), dtype=torch.float)
            )

    # ── Per-layer statistics ──────────────────────────────────────────────────
    # all_stats[layer_idx] = (rel_dist_correct, rel_dist_total)
    # both are dict[relation] -> dict[dist_n] -> int
    all_stats = []

    print(f'\nOpening embeddings file …')
    with h5py.File(emb_path, 'r') as hf:
        for layer_idx in range(n_layers):
            print(f'  Layer {layer_idx:2d} / {n_layers - 1} …', flush=True)
            layer_dir = tp_dir / f'layer-{layer_idx:02d}'

            # Load per-layer config for hidden_dim and rank
            layer_cfgs = list(layer_dir.glob('*.yaml'))
            if not layer_cfgs:
                sys.exit(f'No YAML config found in {layer_dir}')
            with open(layer_cfgs[0]) as f:
                lcfg = yaml.safe_load(f)
            hidden_dim = lcfg['model']['hidden_dim']
            rank       = lcfg['probe']['maximum_rank']
            probe_path = layer_dir / lcfg['probe']['params_path']

            # Load and align embeddings for this layer
            embs = []
            for idx in range(n_sentences):
                raw  = torch.tensor(np.array(hf[str(idx)][layer_idx]), dtype=torch.float)
                align = alignments[idx]
                word_feats = align.t() @ raw[1:-1]   # drop [CLS], [SEP]
                embs.append(word_feats)

            # Load probe
            probe = TwoWordPSDProbe(hidden_dim, rank)
            probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
            probe.eval()

            # Inference & accumulation
            rel_dist_correct = defaultdict(lambda: defaultdict(int))
            rel_dist_total   = defaultdict(lambda: defaultdict(int))

            with torch.no_grad():
                for obs, emb in zip(observations, embs):
                    length      = len(obs.sentence)
                    dist_matrix = probe(emb)

                    pred_edges = {
                        tuple(sorted(e))
                        for e in mst_edges(dist_matrix, obs.xpos_sentence)
                    }

                    edge_to_rel = {}
                    for tok_idx in range(length):
                        head_idx = int(obs.head_indices[tok_idx]) - 1
                        if 0 <= head_idx < length:
                            edge_to_rel[tuple(sorted((tok_idx, head_idx)))] = \
                                obs.governance_relations[tok_idx]

                    for tok_idx in range(length):
                        if obs.xpos_sentence[tok_idx] in PUNCT_POS:
                            continue
                        head_idx = int(obs.head_indices[tok_idx]) - 1
                        if not (0 <= head_idx < length):
                            continue
                        if obs.xpos_sentence[head_idx] in PUNCT_POS:
                            continue

                        edge     = tuple(sorted((tok_idx, head_idx)))
                        relation = edge_to_rel.get(edge, 'UNK')
                        dist_n   = abs(tok_idx - head_idx) - 1

                        rel_dist_total[relation][dist_n]   += 1
                        if edge in pred_edges:
                            rel_dist_correct[relation][dist_n] += 1

            all_stats.append((
                {r: dict(d) for r, d in rel_dist_correct.items()},
                {r: dict(d) for r, d in rel_dist_total.items()},
            ))

    # ── Collect all relations ─────────────────────────────────────────────────
    all_rels = set()
    for _, total in all_stats:
        all_rels |= set(total.keys())
    relations = sorted(all_rels)
    n_rel     = len(relations)
    print(f'\n{n_rel} relations found across all layers')
    print(f'Building figure with {n_layers} × {n_rel} × 2 = {n_layers * n_rel * 2} traces …')

    # ── Build Plotly traces ───────────────────────────────────────────────────
    # Trace layout: for layer L and relation index R:
    #   UAS trace  = traces[2 * (L * n_rel + R)]      x = ['Total', 'n=0', ...]
    #   freq trace = traces[2 * (L * n_rel + R) + 1]  x = ['n=0', 'n=1', ...]
    #
    # barmode='group': at each n-bucket two bars stand side by side —
    #   coral = UAS_n,  steelblue = |S_n|/total.
    # The 'Total' bar (overall UAS for the relation) appears as a single bar
    # because the freq trace has no 'Total' entry.
    traces = []

    for layer_idx in range(n_layers):
        rel_correct, rel_total = all_stats[layer_idx]

        for rel_idx, rel in enumerate(relations):
            dist_counts    = rel_total.get(rel, {})
            correct_counts = rel_correct.get(rel, {})
            all_ns         = sorted(dist_counts.keys())

            total_rel   = sum(dist_counts.values())
            correct_rel = sum(correct_counts.get(n, 0) for n in all_ns)
            total_uuas  = correct_rel / total_rel if total_rel > 0 else 0.0

            n_labels  = [f'n={n}' for n in all_ns]
            uas_vals  = [correct_counts.get(n, 0) / dist_counts[n] for n in all_ns]
            freq_vals = [dist_counts[n] / total_rel if total_rel > 0 else 0.0
                         for n in all_ns]
            counts    = [dist_counts[n] for n in all_ns]

            visible = (layer_idx == 0 and rel_idx == 0)

            # ── UAS trace: 'Total' bar + one coral bar per n ──────────────
            traces.append(go.Bar(
                x=['Total'] + n_labels,
                y=[total_uuas] + uas_vals,
                visible=visible,
                showlegend=False,
                marker_color=['steelblue'] + ['coral'] * len(all_ns),
                customdata=[[correct_rel, total_rel]]
                           + list(zip([correct_counts.get(n, 0) for n in all_ns],
                                      counts)),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    'UAS: %{y:.3f}<br>'
                    'correct: %{customdata[0]}<br>'
                    'total: %{customdata[1]}'
                    '<extra></extra>'
                ),
            ))

            # ── Frequency trace: one steelblue bar per n (no 'Total') ─────
            traces.append(go.Bar(
                x=n_labels,
                y=freq_vals,
                visible=visible,
                showlegend=False,
                marker_color='#6baed6',
                customdata=list(zip(counts, [total_rel] * len(all_ns))),
                hovertemplate=(
                    '<b>%{x}</b><br>'
                    '|S_n| / total: %{y:.3f}<br>'
                    '|S_n|: %{customdata[0]}<br>'
                    'total: %{customdata[1]}'
                    '<extra></extra>'
                ),
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='group',
        title=dict(
            text=f'Labeled UAS by linear distance — Checkpoint 0, <b>{relations[0]}</b>',
            x=0.5, xanchor='center',
            y=0.97, yanchor='top',
        ),
        xaxis_title='n (words between the two endpoints)',
        yaxis_title='value',
        yaxis=dict(range=[0, 1.05], tickformat='.2f'),
        height=580,
        bargap=0.15,
        bargroupgap=0.05,
        font=dict(size=13),
        margin=dict(t=80, r=20),
    )

    # ── Write HTML with custom JS dropdowns ───────────────────────────────────
    print('Writing HTML …')
    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs=True,
        div_id='plotly-graph',
    )

    layer_options = '\n'.join(
        f'        <option value="{i}">Checkpoint {i}</option>'
        for i in range(n_layers)
    )
    rel_options = '\n'.join(
        f'        <option value="{i}">{rel}</option>'
        for i, rel in enumerate(relations)
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>UUAS by Relation × Distance — All Checkpoints</title>
  <style>
    body {{ font-family: sans-serif; margin: 0; padding: 10px; }}
    .controls {{
      display: flex; gap: 24px; align-items: center;
      padding: 10px 14px; background: #f0f0f0;
      border-radius: 5px; margin-bottom: 8px;
    }}
    label {{ font-size: 14px; font-weight: 600; }}
    select {{ font-size: 14px; padding: 4px 8px; margin-left: 6px; }}
  </style>
</head>
<body>
  <div class="controls">
    <label>Checkpoint:
      <select id="layer-select">
{layer_options}
      </select>
    </label>
    <label>Relation:
      <select id="rel-select">
{rel_options}
      </select>
    </label>
  </div>

  {plot_div}

  <script>
    (function() {{
      var gd      = document.getElementById('plotly-graph');
      var nLayers = {n_layers};
      var nRel    = {n_rel};
      var relNames = {json.dumps(relations)};

      function updateViz() {{
        var layerIdx = parseInt(document.getElementById('layer-select').value);
        var relIdx   = parseInt(document.getElementById('rel-select').value);
        var base     = 2 * (layerIdx * nRel + relIdx);

        var visibility = new Array(nLayers * nRel * 2).fill(false);
        visibility[base]     = true;
        visibility[base + 1] = true;

        Plotly.restyle(gd, {{visible: visibility}});
        Plotly.relayout(gd, {{
          'title.text': 'Labeled UAS by linear distance — Checkpoint '
                        + layerIdx + ', <b>' + relNames[relIdx] + '</b>'
        }});
      }}

      document.getElementById('layer-select').addEventListener('change', updateViz);
      document.getElementById('rel-select').addEventListener('change', updateViz);
    }})();
  </script>
</body>
</html>
"""

    out_path.write_text(html, encoding='utf-8')
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
