#!/usr/bin/env python3
"""
Compute UUAS by dependency relation on dev and test for every trained layer.

Skips loading training embeddings (39k sentences) and uses map_location='cpu'
so it runs on CPU-only SLURM nodes.

Usage (from experiments/bert-base-prd/ or anywhere):
    python compute_uuas_by_relation_dev_test.py [--layer N]
    (omit --layer to process all 26 layers 0-25)
    (--force to recompute even if output files already exist)
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(REPO_ROOT, 'structural-probes'))

import data as spdata
import probe as sprobe
import task as sptask
from reporter import prims_matrix_to_edges


class _DevTestOnlyBERTDataset(spdata.BERTDataset):
    """BERTDataset that loads only dev and test, skipping train."""

    def read_from_disk(self):
        cfg        = self.args['dataset']
        corpus_root = cfg['corpus']['root']
        emb_root    = cfg['embeddings']['root']
        align_root  = self.args['model']['alignment_root']

        dev_obs  = self.load_conll_dataset(os.path.join(corpus_root, cfg['corpus']['dev_path']))
        test_obs = self.load_conll_dataset(os.path.join(corpus_root, cfg['corpus']['test_path']))

        dev_emb  = os.path.join(emb_root,   cfg['embeddings']['dev_path'])
        test_emb = os.path.join(emb_root,   cfg['embeddings']['test_path'])
        dev_aln  = os.path.join(align_root, self.args['model']['alignment_dev_path'])
        test_aln = os.path.join(align_root, self.args['model']['alignment_test_path'])

        dev_obs  = self.optionally_add_embeddings(dev_obs,  dev_emb,  dev_aln)
        test_obs = self.optionally_add_embeddings(test_obs, test_emb, test_aln)
        return [], dev_obs, test_obs


def _predict(probe, dataloader):
    batches = []
    with torch.no_grad():
        for observation_batch, label_batch, length_batch, _ in tqdm(dataloader, desc='[predicting]'):
            preds = probe(observation_batch)  # DiskModel: identity; probe: pairwise distances
            batches.append(preds.cpu().numpy())
    return batches


def _write_uuas_by_relation(prediction_batches, dataloader, out_path):
    relation_correct = defaultdict(int)
    relation_total   = defaultdict(int)

    for pred_batch, (_, label_batch, length_batch, obs_batch) in tqdm(
            zip(prediction_batches, dataloader), desc='[uuas_by_relation]'):
        for pred, label, length, (observation, _) in zip(
                pred_batch, label_batch, length_batch, obs_batch):
            words  = observation.sentence
            poses  = observation.xpos_sentence
            length = int(length)
            pred   = pred[:length, :length]
            label  = label[:length, :length].cpu()

            gold_edges     = prims_matrix_to_edges(label, words, poses)
            pred_edges     = prims_matrix_to_edges(pred,  words, poses)
            pred_edges_set = {tuple(sorted(e)) for e in pred_edges}

            edge_to_relation = {}
            for tok_idx in range(length):
                head_idx = int(observation.head_indices[tok_idx]) - 1
                if head_idx < 0 or head_idx >= length:
                    continue
                edge_to_relation[tuple(sorted((tok_idx, head_idx)))] = \
                    observation.governance_relations[tok_idx]

            for i, j in gold_edges:
                edge     = tuple(sorted((i, j)))
                relation = edge_to_relation.get(edge, 'UNK')
                relation_total[relation]   += 1
                if edge in pred_edges_set:
                    relation_correct[relation] += 1

    with open(out_path, 'w') as fout:
        fout.write('relation\tuuas\tcorrect\ttotal\n')
        for rel in sorted(relation_total):
            tot  = relation_total[rel]
            corr = relation_correct[rel]
            fout.write('{}\t{:.4f}\t{}\t{}\n'.format(rel, corr / tot, corr, tot))
    print('Wrote', out_path)


def process_layer(layer_idx, results_dir, force=False):
    layer_pad    = f'{layer_idx:02d}'
    layer_result = os.path.join(results_dir, f'layer-{layer_pad}')
    params_path  = os.path.join(layer_result, 'predictor.params')

    if not os.path.exists(params_path):
        print(f'[layer {layer_pad}] No predictor.params found; skipping.')
        return

    dev_out  = os.path.join(layer_result, 'dev.uuas_by_relation')
    test_out = os.path.join(layer_result, 'test.uuas_by_relation')
    if not force and os.path.exists(dev_out) and os.path.exists(test_out):
        print(f'[layer {layer_pad}] Output files already exist; skipping (use --force to recompute).')
        return

    # Use the per-layer yaml saved during training (has real paths already filled in).
    yaml_candidates = glob.glob(os.path.join(layer_result, 'probe_bertbase_prealigned_layer*.yaml'))
    if not yaml_candidates:
        print(f'[layer {layer_pad}] No layer yaml found in {layer_result}; skipping.')
        return
    layer_yaml = yaml_candidates[0]

    print(f'\n=== Layer {layer_pad} ===')
    cfg = yaml.safe_load(open(layer_yaml))
    cfg['reporting']['root'] = layer_result
    cfg['device'] = torch.device('cpu')

    probe_task = sptask.ParseDistanceTask()
    dataset    = _DevTestOnlyBERTDataset(cfg, probe_task)

    expt_probe = sprobe.TwoWordPSDProbe(cfg)
    expt_probe.load_state_dict(torch.load(params_path, map_location='cpu'))
    expt_probe.eval()

    for split_name, out_path in [('dev', dev_out), ('test', test_out)]:
        if not force and os.path.exists(out_path):
            print(f'  {split_name}.uuas_by_relation already exists; skipping.')
            continue
        dataloader = (dataset.get_dev_dataloader() if split_name == 'dev'
                      else dataset.get_test_dataloader())
        preds = _predict(expt_probe, dataloader)
        dataloader = (dataset.get_dev_dataloader() if split_name == 'dev'
                      else dataset.get_test_dataloader())
        _write_uuas_by_relation(preds, dataloader, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer', type=int, default=-1,
                        help='Layer index (0-25). Omit to process all layers.')
    parser.add_argument('--force', action='store_true',
                        help='Recompute even if output files already exist.')
    args = parser.parse_args()

    results_dir = os.path.join(SCRIPT_DIR, 'results-hface')
    layers = range(26) if args.layer < 0 else [args.layer]
    for layer_idx in layers:
        process_layer(layer_idx, results_dir, force=args.force)


if __name__ == '__main__':
    main()
