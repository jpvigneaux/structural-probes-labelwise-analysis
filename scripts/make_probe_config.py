#!/usr/bin/env python3
"""Write the run_experiment.py config for one model at one checkpoint.

Previously each driver built this inline with a heredoc, which meant every model
carried its own near-copy of the same YAML surgery and they could drift. The
probe hyper-parameters (rank 64, L1 loss, 40 epochs, batch 20) come from the base
config and are identical for every model in the paper -- that identity is what
makes the cross-model comparison like-for-like, so it is worth having one place
that guarantees it.

Only five things vary between runs: the embedding directory, the alignment
directory, the file stem, the hidden dimension, and which checkpoint to read.
All five come from experiments/paper_runs.yaml.

`hidden_dim` is taken from the manifest rather than guessed from the model name,
because it cannot be inferred: BERT-base, GPT-2, DeBERTa-v3-base and
ModernBERT-base are all 768-dimensional, and GPT-J is 4096.

Usage:
    python scripts/make_probe_config.py --run gpt2 --layer 16 \\
        --embeddings-dir .../gpt2 --alignments-dir .../gpt2Tok \\
        --corpus .../training-data --out layer-16.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _manifest                                    # noqa: E402

MANIFEST = _manifest.MANIFEST
BASE = HERE.parent / 'experiments' / 'bert-base-prd' / 'base_config_hface_prealigned.yaml'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--layer', type=int, required=True)
    ap.add_argument('--embeddings-dir', required=True)
    ap.add_argument('--alignments-dir', required=True)
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--base-config', default=str(BASE))
    args = ap.parse_args()

    man = _manifest.load(args.manifest)
    run = man['runs'][args.run]
    ex = run['extraction']

    n_ck = int(ex['n_checkpoints'])
    if not 0 <= args.layer < n_ck:
        raise SystemExit(
            f'checkpoint {args.layer} is outside the {n_ck} this run has '
            f'({ex["layout"]} layout). Submitting a wider array than the model has '
            f'checkpoints would train probes on data that does not exist.')

    cfg = yaml.safe_load(open(args.base_config))
    cfg['model'].update(model_type='hf-disk',
                        hf_model_name=ex['tokenizer'],
                        model_layer=args.layer,
                        hidden_dim=int(ex['hidden_dim']),
                        alignment='pre-aligned',
                        alignment_root=str(args.alignments_dir).rstrip('/') + '/')
    cfg['dataset']['corpus']['root'] = str(args.corpus).rstrip('/') + '/'
    cfg['dataset']['embeddings']['root'] = str(args.embeddings_dir).rstrip('/') + '/'
    for split in ('train', 'dev', 'test'):
        cfg['dataset']['embeddings'][f'{split}_path'] = \
            f'raw.{split}.{ex["stem"]}.hdf5'

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f'{run["label"]} checkpoint {args.layer} -> {args.out}')


if __name__ == '__main__':
    main()
