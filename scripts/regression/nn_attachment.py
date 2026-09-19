#!/usr/bin/env python3
"""Decoder-free per-relation scores: does the tree decoder cost long edges?

UASL scores a gold edge as recovered when it appears in the minimum spanning tree (MST)
of the probe's predicted distances. Because the MST must connect every word, a long
edge can be lost to the decoder even when the probe places the two words close. This
script scores the same gold edges (the ones UASL scores: both endpoints non-punctuation,
labelled by the dependent's relation) without any tree decoding:

  nn_dep   the head is the dependent's nearest neighbour in probe space, among the
           sentence's non-punctuation words
  nn_any   either endpoint is the other's nearest neighbour
  rr_dep   reciprocal rank of the head among the dependent's candidates (graded
           version of nn_dep: 1 if nearest, 1/2 if second, ...)

and, for comparison, `mst` = the UASL criterion itself. Written per relation (same
columns as dev.uuas_by_relation, so the regression scripts read them unchanged) and per
linear distance |i - j|.

    python nn_attachment.py --results-dir RESULTS --layer 16

Writes into RESULTS/layer-NN/: {dev,test}.{mst,nn_dep,nn_any,rr_dep}_by_relation and
{dev,test}.decoder_by_distance.
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'structural-probes'))

from compute_uuas_by_relation import _DevTestOnlyDataset, _predict   # noqa: E402
import probe as sprobe                                                # noqa: E402
import task as sptask                                                 # noqa: E402
from reporter import prims_matrix_to_edges                            # noqa: E402

PUNCT = {"''", ",", ".", ":", "``", "-LRB-", "-RRB-"}
CRITERIA = ('mst', 'nn_dep', 'nn_any', 'rr_dep')


def score(pred_batches, loader):
    by_rel = {c: (defaultdict(int), defaultdict(int)) for c in CRITERIA}
    by_dist = {c: (defaultdict(int), defaultdict(int)) for c in CRITERIA}
    for pred_batch, (_, label_batch, length_batch, obs_batch) in zip(pred_batches, loader):
        for pred, label, length, (obs, _) in zip(pred_batch, label_batch, length_batch, obs_batch):
            n = int(length)
            pred = pred[:n, :n]
            label = label[:n, :n].cpu()
            words, poses = obs.sentence, obs.xpos_sentence
            keep = np.array([p not in PUNCT for p in poses])
            gold = prims_matrix_to_edges(label, words, poses)
            mst = {tuple(sorted(e)) for e in prims_matrix_to_edges(pred, words, poses)}

            masked = np.where(keep[None, :], pred, np.inf)
            np.fill_diagonal(masked, np.inf)
            nn = masked.argmin(axis=1)

            head_of = {}
            for t in range(n):
                h = int(obs.head_indices[t]) - 1
                if 0 <= h < n:
                    head_of[t] = h
            for i, j in gold:
                a, b = sorted((i, j))
                if head_of.get(a) == b:
                    dep, head = a, b
                elif head_of.get(b) == a:
                    dep, head = b, a
                else:
                    dep = head = None       # an edge bridging removed punctuation
                rel = obs.governance_relations[dep] if dep is not None else 'UNK'
                dist = b - a
                if dep is not None:
                    # reciprocal rank of the head among the dependent's candidates
                    rank = int((masked[dep] < masked[dep, head]).sum()) + 1
                hit = {
                    'rr_dep': (1.0 / rank) if dep is not None else 0.0,
                    'mst': (a, b) in mst,
                    'nn_dep': dep is not None and nn[dep] == head,
                    'nn_any': nn[a] == b or nn[b] == a,
                }
                for c in CRITERIA:
                    by_rel[c][1][rel] += 1
                    by_rel[c][0][rel] += float(hit[c])
                    by_dist[c][1][dist] += 1
                    by_dist[c][0][dist] += float(hit[c])
    return by_rel, by_dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--layer', type=int, required=True)
    args = ap.parse_args()

    layer_dir = os.path.join(args.results_dir, f'layer-{args.layer:02d}')
    cfg = yaml.safe_load(open(sorted(glob.glob(os.path.join(layer_dir, '*.yaml')))[0]))
    cfg['reporting']['root'] = layer_dir
    cfg['device'] = torch.device('cpu')
    dataset = _DevTestOnlyDataset(cfg, sptask.ParseDistanceTask())
    probe = sprobe.TwoWordPSDProbe(cfg)
    probe.load_state_dict(torch.load(os.path.join(layer_dir, 'predictor.params'), map_location='cpu'))
    probe.eval()

    for split, get in (('dev', dataset.get_dev_dataloader), ('test', dataset.get_test_dataloader)):
        by_rel, by_dist = score(_predict(probe, get()), get())
        for c in CRITERIA:
            correct, total = by_rel[c]
            with open(os.path.join(layer_dir, f'{split}.{c}_by_relation'), 'w') as f:
                f.write('relation\tuuas\tcorrect\ttotal\n')
                for r in sorted(total):
                    f.write(f'{r}\t{correct[r] / total[r]:.4f}\t{correct[r]:g}\t{total[r]}\n')
        with open(os.path.join(layer_dir, f'{split}.decoder_by_distance'), 'w') as f:
            f.write('distance\ttotal\t' + '\t'.join(CRITERIA) + '\n')
            for d in sorted(by_dist['mst'][1]):
                tot = by_dist['mst'][1][d]
                f.write(f'{d}\t{tot}\t' + '\t'.join(f'{by_dist[c][0][d] / tot:.4f}' for c in CRITERIA) + '\n')
        overall = {c: sum(by_rel[c][0].values()) / sum(by_rel[c][1].values()) for c in CRITERIA}
        print(split, {c: round(v, 4) for c, v in overall.items()})


if __name__ == '__main__':
    main()
