#!/usr/bin/env python3
"""
For each PTB dependency relation, partition gold edges by linear distance
(S_n = edges where exactly n words lie between the two endpoints), run the
structural probe on the chosen layer, and produce an interactive Plotly
figure.

The figure has a dropdown that selects the relation.  For each relation the
chart shows:
  - one bar labelled "Total": the overall labeled UUAS for that relation
  - one bar per n = 0, 1, 2, …: UUAS restricted to S_n

Usage (on a Quest CPU node):
    python uuas_by_relation_distance.py [--layer 8] [--split dev] \
        [--experiment-dir /path/to/structural-probes-labelwise-analysis/experiments/bert-base-prd] \
        [--out path/to/output.html]
"""
import argparse
import sys
from collections import defaultdict, namedtuple
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
import plotly.graph_objects as go

# ─── CoNLL-X parser ────────────────────────────────────────────────────────────

_FIELDS = [
    'index', 'sentence', 'lemma_sentence', 'upos_sentence', 'xpos_sentence',
    'morph', 'head_indices', 'governance_relations', 'secondary_relations',
    'extra_info', 'embeddings',
]
Observation = namedtuple('Observation', _FIELDS)


def load_conll(path):
    """Return a list of Observations parsed from a CoNLL-X file."""
    observations = []
    buf = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if not line:
                if buf:
                    cols = list(zip(*[tok.split('\t') for tok in buf]))
                    n = len(cols[0])
                    observations.append(Observation(*cols, [None] * n))
                    buf = []
            else:
                buf.append(line)
    if buf:
        cols = list(zip(*[tok.split('\t') for tok in buf]))
        observations.append(Observation(*cols, [None] * len(cols[0])))
    return observations


# ─── Embedding loader (pre-aligned) ───────────────────────────────────────────

def load_embeddings_prealigned(emb_path, align_path, n_sentences, layer):
    """
    Load pre-aligned BERT embeddings.

    Each sentence key in the HDF5 file holds a tensor of shape
    (n_layers, n_subwords, hidden_dim).  The alignment file holds a matrix
    of shape (n_subwords - 2, n_words) (excluding [CLS] / [SEP]).

    Returns a list of (n_words, hidden_dim) float tensors.
    """
    result = []
    with h5py.File(emb_path, 'r') as hf, h5py.File(align_path, 'r') as af:
        for idx in range(n_sentences):
            raw = torch.tensor(np.array(hf[str(idx)][layer]), dtype=torch.float)  # (n_sub, hidden)
            align = torch.tensor(np.array(af[str(idx)]), dtype=torch.float)        # (n_sub-2, n_words)
            sub_feats = raw[1:-1]                                                   # drop [CLS], [SEP]
            word_feats = align.t() @ sub_feats                                      # (n_words, hidden)
            result.append(word_feats)
    return result


# ─── Structural probe (TwoWordPSDProbe) ───────────────────────────────────────

class TwoWordPSDProbe(nn.Module):
    """Squared-L2 distance probe: d(i,j) = ||B(h_i - h_j)||^2."""

    def __init__(self, hidden_dim, rank):
        super().__init__()
        self.proj = nn.Parameter(torch.zeros(hidden_dim, rank))

    def forward(self, embeddings):
        """embeddings: (n, hidden) → distances: (n, n)"""
        t = embeddings @ self.proj          # (n, rank)
        diff = t.unsqueeze(0) - t.unsqueeze(1)  # (n, n, rank)
        return diff.pow(2).sum(-1)          # (n, n)


# ─── Prim's MST (identical logic to reporter.py) ──────────────────────────────

PUNCT_POS = {"''", ",", ".", ":", "``", "-LRB-", "-RRB-"}


class _UnionFind:
    def __init__(self, n):
        self.parents = list(range(n))

    def find(self, i):
        while self.parents[i] != i:
            i = self.parents[i]
        return i

    def union(self, i, j):
        pi = self.find(i)
        self.parents[pi] = j   # same convention as reporter.py


def mst_edges(dist_matrix, poses):
    """Return list of (i, j) edges forming the MST; punctuation tokens excluded."""
    n = len(poses)
    pairs = {}
    for i in range(n):
        for j in range(n):
            if poses[i] in PUNCT_POS or poses[j] in PUNCT_POS:
                continue
            pairs[(i, j)] = dist_matrix[i, j].item()

    uf = _UnionFind(n)
    edges = []
    for (i, j), _ in sorted(pairs.items(), key=lambda x: x[1]):
        if uf.find(i) != uf.find(j):
            uf.union(i, j)
            edges.append((i, j))
    return edges


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--layer', type=int, default=16,
                   help='BERT layer index (0-25; default: 8)')
    p.add_argument('--split', choices=['dev', 'test'], default='dev',
                   help='Data split to evaluate (default: dev)')
    p.add_argument('--trained-probes-dir', default=str(Path(__file__).parent / 'results-hface'),
                   help='Path to the bert-base-prd experiment directory')
    p.add_argument('--out', default=None,
                   help='Output HTML path (default: <layer_dir>/uuas_by_relation_distance.html)')
    return p.parse_args()


def main():
    probe_dir = Path(__file__).parent 
    args = parse_args()
    tp_dir = Path(args.trained_probes_dir)
    layer_dir = tp_dir  / f'layer-{args.layer:02d}'
    out_path = Path(args.out) if args.out is not None else probe_dir / 'figures' / f'layer-{args.layer:02d}' / 'uuas_by_relation_distance.html'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ── Config ────────────────────────────────────────────────────────────────
    config_files = list(layer_dir.glob('*.yaml'))
    if not config_files:
        sys.exit(f'No YAML config found in {layer_dir}')
    with open(config_files[0]) as f:
        cfg = yaml.safe_load(f)

    split = args.split
    corpus_root = Path(cfg['dataset']['corpus']['root'])
    emb_root    = Path(cfg['dataset']['embeddings']['root'])
    align_root  = Path(cfg['model']['alignment_root'])

    conll_path  = corpus_root / cfg['dataset']['corpus'][f'{split}_path']
    emb_path    = emb_root    / cfg['dataset']['embeddings'][f'{split}_path']
    align_path  = align_root  / cfg['model'][f'alignment_{split}_path']
    probe_path  = layer_dir   / cfg['probe']['params_path']
    layer_idx   = cfg['model']['model_layer']
    hidden_dim  = cfg['model']['hidden_dim']
    rank        = cfg['probe']['maximum_rank']

    print(f'Layer {layer_idx} | split={split}')
    for label, p in [('conll', conll_path), ('embs', emb_path),
                     ('align', align_path), ('probe', probe_path)]:
        print(f'  {label:<6}: {p}')

    # ── Load data ─────────────────────────────────────────────────────────────
    print('Loading CoNLL data …')
    observations = load_conll(conll_path)

    print('Loading embeddings …')
    embeddings_list = load_embeddings_prealigned(
        emb_path, align_path, len(observations), layer_idx)

    # ── Load probe ────────────────────────────────────────────────────────────
    print('Loading probe …')
    probe = TwoWordPSDProbe(hidden_dim, rank)
    probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
    probe.eval()

    # ── Inference & accumulation ───────────────────────────────────────────────
    # rel_dist_correct[relation][n] = number of edges in S_n correctly predicted
    # rel_dist_total  [relation][n] = total gold edges in S_n
    rel_dist_correct = defaultdict(lambda: defaultdict(int))
    rel_dist_total   = defaultdict(lambda: defaultdict(int))

    print('Running inference …')
    with torch.no_grad():
        for obs, emb in zip(observations, embeddings_list):
            length = len(obs.sentence)
            dist_matrix = probe(emb)  # (length, length)

            pred_edges_set = {
                tuple(sorted(e)) for e in mst_edges(dist_matrix, obs.xpos_sentence)
            }

            # Build edge → relation map from the gold parse
            edge_to_relation = {}
            for tok_idx in range(length):
                head_idx = int(obs.head_indices[tok_idx]) - 1  # 0-indexed
                if head_idx < 0 or head_idx >= length:
                    continue
                edge = tuple(sorted((tok_idx, head_idx)))
                edge_to_relation[edge] = obs.governance_relations[tok_idx]

            # Iterate over gold edges (same punctuation filter as Prim's)
            for tok_idx in range(length):
                if obs.xpos_sentence[tok_idx] in PUNCT_POS:
                    continue
                head_idx = int(obs.head_indices[tok_idx]) - 1
                if head_idx < 0 or head_idx >= length:
                    continue
                if obs.xpos_sentence[head_idx] in PUNCT_POS:
                    continue

                edge     = tuple(sorted((tok_idx, head_idx)))
                relation = edge_to_relation.get(edge, 'UNK')
                dist_n   = abs(tok_idx - head_idx) - 1  # words in between

                rel_dist_total[relation][dist_n] += 1
                if edge in pred_edges_set:
                    rel_dist_correct[relation][dist_n] += 1

    # ── Build Plotly figure ───────────────────────────────────────────────────
    print('Building figure …')
    relations = sorted(rel_dist_total.keys())

    # Two traces per relation: base (UUAS per bucket) + contribution overlay.
    # Total number of traces = 2 * len(relations).
    fig = go.Figure()

    for i, rel in enumerate(relations):
        dist_counts = rel_dist_total[rel]
        all_ns      = sorted(dist_counts.keys())

        total_rel   = sum(dist_counts[n] for n in all_ns)
        correct_rel = sum(rel_dist_correct[rel].get(n, 0) for n in all_ns)
        total_uuas  = correct_rel / total_rel if total_rel > 0 else 0.0

        x_labels   = ['Total'] + [f'n={n}' for n in all_ns]
        uuas_vals  = [total_uuas] + [
            rel_dist_correct[rel].get(n, 0) / dist_counts[n]
            for n in all_ns
        ]
        # contribution_n = UUAS_n × (total_n / total_rel) = correct_n / total_rel
        # "Total" bar gets 0 so it is not double-stacked.
        contrib_vals = [0.0] + [
            rel_dist_correct[rel].get(n, 0) / total_rel
            for n in all_ns
        ]
        counts        = [total_rel] + [dist_counts[n] for n in all_ns]
        correct_vals  = [correct_rel] + [rel_dist_correct[rel].get(n, 0) for n in all_ns]
        visible       = (i == 0)

        # ── base bars (UUAS per bucket) ──────────────────────────────────────
        fig.add_trace(go.Bar(
            x=x_labels,
            y=uuas_vals,
            name=rel,
            visible=visible,
            showlegend=False,
            marker_color=['steelblue'] + ['coral'] * len(all_ns),
            customdata=list(zip(counts, correct_vals, contrib_vals)),
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Labeled UAS: %{y:.3f}<br>'
                'correct: %{customdata[1]}<br>'
                'total: %{customdata[0]}'
                '<extra></extra>'
            ),
        ))

        # ── contribution overlay (darker orange, superimposed) ──────────────
        # Drawn after the base trace so it renders on top.
        # Height = correct_n / total_rel ≤ UUAS_n, so it always fits inside.
        # "Total" bar is transparent (no overlay needed).
        fig.add_trace(go.Bar(
            x=x_labels,
            y=contrib_vals,
            name=rel,
            visible=visible,
            showlegend=False,
            marker_color=['rgba(0,0,0,0)'] + ['#b84c00'] * len(all_ns),
            customdata=list(zip(counts, correct_vals, uuas_vals)),
            hovertemplate=(
                '<b>%{x}</b> — contribution to total Labeled UAS<br>'
                'Labeled UAS_n × |S_n|/total: %{y:.3f}<br>'
                'Labeled UAS_n: %{customdata[2]:.3f}<br>'
                '|S_n|: %{customdata[0]} / ' + str(total_rel) + '<br>'
                'correct: %{customdata[1]}'
                '<extra></extra>'
            ),
        ))

    # Each dropdown button reveals exactly 2 traces (base + overlay).
    n_rel = len(relations)
    buttons = [
        dict(
            label=rel,
            method='update',
            args=[
                {'visible': [
                    j == 2 * i or j == 2 * i + 1
                    for j in range(2 * n_rel)
                ]},
                # 'title.text' is the correct key for layout title updates in buttons
                {'title.text': f'Labeled UAS by linear distance — <b>{rel}</b>'},
            ],
        )
        for i, rel in enumerate(relations)
    ]

    first_rel = relations[0]
    fig.update_layout(
        barmode='overlay',
        title=dict(
            text=f'Labeled UAS by linear distance — <b>{first_rel}</b>',
            x=0.5, xanchor='center',
            y=0.97, yanchor='top',
        ),
        xaxis_title='n (words between the two endpoints)',
        yaxis_title='Labeled UAS',
        yaxis=dict(range=[0, 1.12], tickformat='.2f'),
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            showactive=True,
            x=1.0, xanchor='right',
            y=1.12, yanchor='top',
        )],
        height=580,
        bargap=0.2,
        font=dict(size=13),
        margin=dict(t=80, r=160),
    )

    fig.write_html(out_path)
    print(f'Saved → {out_path}')


if __name__ == '__main__':
    main()
