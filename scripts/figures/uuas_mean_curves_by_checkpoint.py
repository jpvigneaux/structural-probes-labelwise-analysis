#!/usr/bin/env python3
"""
For each selected checkpoint (default: 3, 5, 7, …, 25 — the post-block
residuals, analogous to the layers in Clark et al.), compute:

  • For every dependency relation r and distance n:
      UAS_n(r)  — labeled UAS restricted to edges in S_n
      (only kept when |S_n(r)| >= MIN_COUNT, default 5)

  • Aggregate over relations at each distance n:
      mean_n  = mean  of { UAS_n(r) : |S_n(r)| >= MIN_COUNT }
      std_n   = stdev of { UAS_n(r) : |S_n(r)| >= MIN_COUNT }

Produces a single HTML with one line per checkpoint (colored by depth)
and a semitransparent ±1-std band of the same color.

Typical invocation (Quest CPU node):
    python uuas_mean_curves_by_checkpoint.py \
        [--trained-probes-dir results-hface]
        [--out figures/uuas_mean_curves_by_checkpoint.html]
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
import plotly.colors as pc

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


# ── Prim's MST ────────────────────────────────────────────────────────────────

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
        for i in range(n) for j in range(n)
        if poses[i] not in PUNCT_POS and poses[j] not in PUNCT_POS
    }
    uf = _UnionFind(n)
    edges = []
    for (i, j), _ in sorted(pairs.items(), key=lambda x: x[1]):
        if uf.find(i) != uf.find(j):
            uf.union(i, j)
            edges.append((i, j))
    return edges


# ── Helpers ───────────────────────────────────────────────────────────────────

def rgb_to_rgba(rgb_str, alpha):
    r, g, b = pc.unlabel_rgb(rgb_str)
    return f'rgba({r},{g},{b},{alpha})'


# ── Argument parsing ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trained-probes-dir',
                   default=str(Path(__file__).parent / 'results-hface'),
                   help='Directory containing layer-00 … layer-25 subdirs')
    p.add_argument('--checkpoints', type=int, nargs='+',
                   default=list(range(3, 26, 2)),
                   help='Checkpoint indices to plot (default: 3 5 7 … 25)')
    p.add_argument('--min-count', type=int, default=5,
                   help='Minimum |S_n(r)| to include a (relation, distance) pair (default: 5)')
    p.add_argument('--min-relations', type=int, default=3,
                   help='Minimum number of relations required at a distance to plot it (default: 3)')
    p.add_argument('--out', default=None,
                   help='Output HTML path (default: figures/uuas_mean_curves_by_checkpoint.html)')
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    tp_dir    = Path(args.trained_probes_dir)
    script_dir = Path(__file__).parent
    out_path  = (
        Path(args.out) if args.out
        else script_dir / 'figures' / 'uuas_mean_curves_by_checkpoint.html'
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoints = sorted(args.checkpoints)
    min_count    = args.min_count
    min_relations = args.min_relations
    split       = 'test'

    print(f'Checkpoints : {checkpoints}')
    print(f'Min count   : {min_count}')
    print(f'Min rels    : {min_relations}')

    # ── Common paths (from layer-00 config) ──────────────────────────────────
    layer0_dir   = tp_dir / 'layer-00'
    config_files = list(layer0_dir.glob('*.yaml'))
    if not config_files:
        sys.exit(f'No YAML config in {layer0_dir}')
    with open(config_files[0]) as f:
        cfg0 = yaml.safe_load(f)

    corpus_root = Path(cfg0['dataset']['corpus']['root'])
    emb_root    = Path(cfg0['dataset']['embeddings']['root'])
    align_root  = Path(cfg0['model']['alignment_root'])
    conll_path  = corpus_root / cfg0['dataset']['corpus'][f'{split}_path']
    emb_path    = emb_root    / cfg0['dataset']['embeddings'][f'{split}_path']
    align_path  = align_root  / cfg0['model'][f'alignment_{split}_path']

    print(f'CoNLL       : {conll_path}')
    print(f'Embeddings  : {emb_path}')

    # ── Load corpus ───────────────────────────────────────────────────────────
    print('\nLoading CoNLL data …')
    observations = load_conll(conll_path)
    n_sentences  = len(observations)
    print(f'  {n_sentences} sentences')

    # ── Load alignments (layer-independent) ───────────────────────────────────
    print('Loading alignments …')
    alignments = []
    with h5py.File(align_path, 'r') as af:
        # How many special tokens the tokenizer adds is recorded by
        # precompute_alignments_hf.py: 1/1 for BERT-style encoders ([CLS]/[SEP]),
        # but 0/0 for GPT-2 and GPT-J. The previous hard-coded raw[1:-1] assumed
        # 1/1 for every model and so mis-sliced the decoders.
        n_pre = int(af.attrs.get('n_prefix_special', 1))
        n_suf = int(af.attrs.get('n_suffix_special', 1))
        print(f'  stripping {n_pre} leading / {n_suf} trailing special-token rows')
        for idx in range(n_sentences):
            alignments.append(
                torch.tensor(np.array(af[str(idx)]), dtype=torch.float)
            )

    # ── Compute per-checkpoint stats ──────────────────────────────────────────
    # ck_stats[i] = (rel_dist_correct, rel_dist_total) for checkpoints[i]
    ck_stats = []

    print(f'\nOpening embeddings file …')
    with h5py.File(emb_path, 'r') as hf:
        for ck in checkpoints:
            print(f'  Checkpoint {ck:2d} …', flush=True)
            layer_dir  = tp_dir / f'layer-{ck:02d}'
            layer_cfgs = list(layer_dir.glob('*.yaml'))
            if not layer_cfgs:
                sys.exit(f'No YAML config in {layer_dir}')
            with open(layer_cfgs[0]) as f:
                lcfg = yaml.safe_load(f)

            hidden_dim = lcfg['model']['hidden_dim']
            rank       = lcfg['probe']['maximum_rank']
            probe_path = layer_dir / lcfg['probe']['params_path']

            # Load word embeddings for this checkpoint
            embs = []
            for idx in range(n_sentences):
                raw        = torch.tensor(np.array(hf[str(idx)][ck]), dtype=torch.float)
                end        = raw.shape[0] - n_suf
                word_feats = alignments[idx].t() @ raw[n_pre:end]
                embs.append(word_feats)

            # Load probe
            probe = TwoWordPSDProbe(hidden_dim, rank)
            probe.load_state_dict(torch.load(probe_path, map_location='cpu'))
            probe.eval()

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

            ck_stats.append((
                {r: dict(d) for r, d in rel_dist_correct.items()},
                {r: dict(d) for r, d in rel_dist_total.items()},
            ))

    # ── Aggregate: mean ± std over relations per (checkpoint, distance) ───────
    print('\nAggregating …')

    # Gather all (n, relation) pairs that pass the min-count filter in any checkpoint
    all_relations = set()
    all_ns_global = set()
    for rel_correct, rel_total in ck_stats:
        for r, counts in rel_total.items():
            all_relations.add(r)
            for n, c in counts.items():
                if c >= min_count:
                    all_ns_global.add(n)

    all_ns = sorted(all_ns_global)
    relations = sorted(all_relations)
    print(f'  {len(relations)} relations,  distances: {min(all_ns)}–{max(all_ns)}')

    # For each checkpoint compute (xs, means, stds, n_rels) arrays
    ck_curves = []
    for ci, ck in enumerate(checkpoints):
        rel_correct, rel_total = ck_stats[ci]
        xs, means, stds, n_rels_used = [], [], [], []
        for n in all_ns:
            uas_list = []
            for r in relations:
                cnt = rel_total.get(r, {}).get(n, 0)
                if cnt >= min_count:
                    uas_list.append(rel_correct.get(r, {}).get(n, 0) / cnt)
            if len(uas_list) >= min_relations:
                xs.append(n)
                means.append(float(np.mean(uas_list)))
                stds.append(float(np.std(uas_list, ddof=0)))
                n_rels_used.append(len(uas_list))
        ck_curves.append((np.array(xs), np.array(means), np.array(stds),
                          np.array(n_rels_used)))

    # ── Build Plotly figure ───────────────────────────────────────────────────
    print('Building figure …')
    n_ck   = len(checkpoints)
    # Sequential palette: purple (early) → yellow-green (late) via Viridis
    palette = pc.sample_colorscale('Viridis', np.linspace(0.05, 0.92, n_ck))

    fig = go.Figure()

    for ci, ck in enumerate(checkpoints):
        xs, means, stds, n_rels = ck_curves[ci]
        if len(xs) == 0:
            continue
        color      = palette[ci]
        fill_color = rgb_to_rgba(color, 0.15)

        upper = means + stds
        lower = np.maximum(means - stds, 0.0)

        # Upper bound — invisible line, used as the ceiling for fill
        fig.add_trace(go.Scatter(
            x=xs, y=upper,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))

        # Lower bound — fill area between this trace and the one above
        fig.add_trace(go.Scatter(
            x=xs, y=lower,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor=fill_color,
            showlegend=False,
            hoverinfo='skip',
        ))

        # Mean line with markers
        fig.add_trace(go.Scatter(
            x=xs,
            y=means,
            mode='lines+markers',
            name=f'Checkpoint {ck}',
            line=dict(color=color, width=2),
            marker=dict(size=5, color=color),
            customdata=np.stack([stds, n_rels], axis=1),
            hovertemplate=(
                f'<b>Checkpoint {ck}</b><br>'
                'distance n = %{x}<br>'
                'mean UAS = %{y:.3f}<br>'
                'std = %{customdata[0]:.3f}<br>'
                '# relations = %{customdata[1]}'
                '<extra></extra>'
            ),
        ))

    fig.update_layout(
        title=dict(
            text='Mean labeled UAS by distance — per checkpoint (test set)',
            x=0.5, xanchor='center',
            y=0.97, yanchor='top',
        ),
        xaxis=dict(
            title='n (words between the two endpoints)',
            dtick=1,
        ),
        yaxis=dict(
            title='Mean labeled UAS across relations',
            range=[0, 1.02],
            tickformat='.2f',
        ),
        legend=dict(
            title='Checkpoint',
            x=1.01, xanchor='left',
            y=1.0,  yanchor='top',
        ),
        height=560,
        font=dict(size=13),
        margin=dict(t=80, r=160),
        hovermode='x unified',
    )

    fig.write_html(out_path, include_plotlyjs=True)
    print(f'Saved → {out_path}')

    # ── Save curve data (aggregated + per-relation) ───────────────────────────
    npz_path  = out_path.with_suffix('.npz')
    save_data = {'checkpoints': np.array(checkpoints),
                 'rel_names':   np.array(relations)}

    # Aggregated mean/std curves
    for ci in range(len(checkpoints)):
        xs, means, stds, n_rels = ck_curves[ci]
        save_data[f'xs_{ci}']    = xs
        save_data[f'means_{ci}'] = means
        save_data[f'stds_{ci}']  = stds
        save_data[f'nrels_{ci}'] = n_rels

    # Raw per-relation (ns, uas, counts) arrays for clustering.
    # Counts come from the gold corpus and are checkpoint-independent;
    # we read them from checkpoint 0 and save once under 'rel_counts_{ri}'.
    # Per-checkpoint (ns, uas) arrays are also saved (min_count filter applied).
    _, rel_total_ref = ck_stats[0]   # gold counts — same for every checkpoint

    for ri, rel in enumerate(relations):
        all_ns_for_rel = sorted(rel_total_ref.get(rel, {}).keys())
        if all_ns_for_rel:
            counts_arr = np.array(
                [rel_total_ref[rel][n] for n in all_ns_for_rel], dtype=np.int32
            )
            all_ns_arr = np.array(all_ns_for_rel, dtype=np.int32)
        else:
            counts_arr = np.array([], dtype=np.int32)
            all_ns_arr = np.array([], dtype=np.int32)
        save_data[f'rel_all_ns_{ri}']    = all_ns_arr
        save_data[f'rel_counts_{ri}']    = counts_arr   # |S_n(r)|, all n

    for ci, _ck in enumerate(checkpoints):
        rel_correct, rel_total = ck_stats[ci]
        for ri, rel in enumerate(relations):
            total_d   = rel_total.get(rel, {})
            correct_d = rel_correct.get(rel, {})
            valid_ns  = sorted(n for n, cnt in total_d.items()
                               if cnt >= min_count)
            if valid_ns:
                ns_arr  = np.array(valid_ns, dtype=np.int32)
                uas_arr = np.array(
                    [correct_d.get(n, 0) / total_d[n] for n in valid_ns]
                )
            else:
                ns_arr  = np.array([], dtype=np.int32)
                uas_arr = np.array([], dtype=np.float64)
            save_data[f'rel_ns_{ci}_{ri}']  = ns_arr
            save_data[f'rel_uas_{ci}_{ri}'] = uas_arr

    np.savez(npz_path, **save_data)
    print(f'Curves data → {npz_path}')


if __name__ == '__main__':
    main()
