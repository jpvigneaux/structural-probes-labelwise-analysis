#!/usr/bin/env python3
"""Compute per-relation UUAS (UASL) on dev and test for trained probe checkpoints.

Generalised from an earlier BERT-only version:
that version hard-coded BERT-base's results directory and config filename, so it
could not be pointed at the other models. Here the results directory and the
checkpoint list are arguments, and the per-checkpoint YAML is discovered by glob,
which works for every model this repo trains.

Training embeddings (39k sentences) are never loaded and everything runs on CPU,
so this is a cheap CPU-only job even for GPT-J.

Writes, into each `<results-dir>/layer-NN/`:
    dev.uuas_by_relation     relation, uuas, correct, total
    test.uuas_by_relation

Usage:
    python compute_uuas_by_relation.py --results-dir RESULTS --layers 0-25
    python compute_uuas_by_relation.py --results-dir RESULTS --layers 16 --force
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'structural-probes'))

import data as spdata
import probe as sprobe
import task as sptask
from reporter import prims_matrix_to_edges


class _DevTestOnlyDataset(spdata.BERTDataset):
    """BERTDataset that loads only dev and test, skipping the train split."""

    def read_from_disk(self):
        cfg = self.args['dataset']
        corpus_root = cfg['corpus']['root']
        emb_root = cfg['embeddings']['root']
        align_root = self.args['model']['alignment_root']

        dev_obs = self.load_conll_dataset(
            os.path.join(corpus_root, cfg['corpus']['dev_path']))
        test_obs = self.load_conll_dataset(
            os.path.join(corpus_root, cfg['corpus']['test_path']))

        dev_emb = os.path.join(emb_root, cfg['embeddings']['dev_path'])
        test_emb = os.path.join(emb_root, cfg['embeddings']['test_path'])
        dev_aln = os.path.join(align_root, self.args['model']['alignment_dev_path'])
        test_aln = os.path.join(align_root, self.args['model']['alignment_test_path'])

        dev_obs = self.optionally_add_embeddings(dev_obs, dev_emb, dev_aln)
        test_obs = self.optionally_add_embeddings(test_obs, test_emb, test_aln)
        return [], dev_obs, test_obs


def _predict(probe, dataloader):
    batches = []
    with torch.no_grad():
        for observation_batch, _, _, _ in tqdm(dataloader, desc='[predicting]'):
            batches.append(probe(observation_batch).cpu().numpy())
    return batches


def _write_uuas_by_relation(prediction_batches, dataloader, out_path):
    relation_correct = defaultdict(int)
    relation_total = defaultdict(int)

    for pred_batch, (_, label_batch, length_batch, obs_batch) in tqdm(
            zip(prediction_batches, dataloader), desc='[uuas_by_relation]'):
        for pred, label, length, (observation, _) in zip(
                pred_batch, label_batch, length_batch, obs_batch):
            words = observation.sentence
            poses = observation.xpos_sentence
            length = int(length)
            pred = pred[:length, :length]
            label = label[:length, :length].cpu()

            gold_edges = prims_matrix_to_edges(label, words, poses)
            pred_edges = prims_matrix_to_edges(pred, words, poses)
            pred_edges_set = {tuple(sorted(e)) for e in pred_edges}

            edge_to_relation = {}
            for tok_idx in range(length):
                head_idx = int(observation.head_indices[tok_idx]) - 1
                if head_idx < 0 or head_idx >= length:
                    continue
                edge_to_relation[tuple(sorted((tok_idx, head_idx)))] = \
                    observation.governance_relations[tok_idx]

            for i, j in gold_edges:
                edge = tuple(sorted((i, j)))
                relation = edge_to_relation.get(edge, 'UNK')
                relation_total[relation] += 1
                if edge in pred_edges_set:
                    relation_correct[relation] += 1

    with open(out_path, 'w') as fout:
        fout.write('relation\tuuas\tcorrect\ttotal\n')
        for rel in sorted(relation_total):
            tot, corr = relation_total[rel], relation_correct[rel]
            fout.write(f'{rel}\t{corr / tot:.4f}\t{corr}\t{tot}\n')
    print('Wrote', out_path)


def process_layer(layer_idx, results_dir, force=False):
    layer_result = os.path.join(results_dir, f'layer-{layer_idx:02d}')
    params_path = os.path.join(layer_result, 'predictor.params')
    if not os.path.exists(params_path):
        print(f'[layer {layer_idx:02d}] no predictor.params; skipping.')
        return

    dev_out = os.path.join(layer_result, 'dev.uuas_by_relation')
    test_out = os.path.join(layer_result, 'test.uuas_by_relation')
    if not force and os.path.exists(dev_out) and os.path.exists(test_out):
        print(f'[layer {layer_idx:02d}] already done; skipping (--force to redo).')
        return

    # Any *.yaml in the layer directory: run_experiment.py copies the exact
    # config it trained with, so paths are already resolved.
    yamls = [p for p in glob.glob(os.path.join(layer_result, '*.yaml'))]
    if not yamls:
        print(f'[layer {layer_idx:02d}] no config yaml in {layer_result}; skipping.')
        return

    print(f'\n=== layer {layer_idx:02d} ({os.path.basename(yamls[0])}) ===')
    cfg = yaml.safe_load(open(yamls[0]))
    cfg['reporting']['root'] = layer_result
    cfg['device'] = torch.device('cpu')

    dataset = _DevTestOnlyDataset(cfg, sptask.ParseDistanceTask())
    expt_probe = sprobe.TwoWordPSDProbe(cfg)
    expt_probe.load_state_dict(torch.load(params_path, map_location='cpu'))
    expt_probe.eval()

    for split_name, out_path in (('dev', dev_out), ('test', test_out)):
        if not force and os.path.exists(out_path):
            print(f'  {split_name} exists; skipping.')
            continue
        get = (dataset.get_dev_dataloader if split_name == 'dev'
               else dataset.get_test_dataloader)
        preds = _predict(expt_probe, get())
        _write_uuas_by_relation(preds, get(), out_path)


def parse_layers(spec, results_dir):
    if not spec:
        found = sorted(glob.glob(os.path.join(results_dir, 'layer-*')))
        return [int(os.path.basename(p).split('-')[1]) for p in found]
    out = []
    for part in spec.split(','):
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True,
                    help='directory containing layer-NN/ subdirectories')
    ap.add_argument('--layers', default=None,
                    help='e.g. "0-25" or "8,16". Default: every layer-* present.')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    for layer_idx in parse_layers(args.layers, args.results_dir):
        process_layer(layer_idx, args.results_dir, force=args.force)


if __name__ == '__main__':
    main()
