#!/usr/bin/env python3
"""Check that the two extraction paths agree on a post-LayerNorm encoder.

The paper's six runs were not all extracted the same way. Four use the `2+2L`
layout of convert_raw_to_hf_all_checkpoints.py; DeBERTa-v3-base and
RoBERTa-Shuffle-n1 use the `1+L` layout of convert_raw_to_hf_natural_sentences.py,
so "checkpoint 9" means the ninth *block* for those two and something else for
the rest. That is recorded in experiments/paper_runs.yaml, but a reader is
entitled to ask whether the two paths would even produce the same numbers.

For a post-LayerNorm architecture they should, and this script demonstrates it
end to end by running the actual scripts rather than reimplementing them:

    1. convert_raw_to_hf_all_checkpoints.py   -> A, the 2+2L pre-LayerNorm
                                                 accumulators
    2. apply_consuming_layernorm.py           -> B, each checkpoint after the
                                                 LayerNorm its consumer applies
    3. convert_raw_to_hf_natural_sentences.py -> H, hidden_states, 1+L

and then compares B[2k+1] against H[k+1] for every block k, plus B[1] against
H[0]. If these agree, the `1+L` runs are a subset of what the general extractor
produces and nothing about them is unreproducible.

They will not agree bit for bit: step 1 stores in float32 and step 2 recomputes
a LayerNorm, so differences of order 1e-6 are expected and the threshold is set
accordingly. A disagreement far above that means the consumer-LayerNorm map in
apply_consuming_layernorm.py is wrong for this architecture, which is worth
knowing before trusting a convention-B file.

Runs on CPU in a couple of minutes for a base-size encoder on 50 sentences.

Usage:
    python scripts/check_extractor_equivalence.py \\
        --model microsoft/deberta-v3-base \\
        --conllx $DATA_DIR/ptb3-wsj-dev.conllx --n-sentences 50
"""

import argparse
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent


def take_sentences(src, dst, n):
    """Copy the first n blank-line-delimited sentences of a CoNLL-X file."""
    out, seen = [], 0
    for line in open(src):
        if not line.strip():
            seen += 1
            out.append(line)
            if seen >= n:
                break
        else:
            out.append(line)
    Path(dst).write_text(''.join(out))
    return seen


def run(cmd):
    print('  $ ' + ' '.join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f'failed: {cmd[1]}')
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='post-LayerNorm encoder, e.g. microsoft/deberta-v3-base')
    ap.add_argument('--model-type', default=None,
                    help='HF model_type for apply_consuming_layernorm.py '
                         '(default: read from the model config)')
    ap.add_argument('--conllx', required=True)
    ap.add_argument('--n-sentences', type=int, default=50)
    ap.add_argument('--workdir', default=None,
                    help='where to put the three HDF5 files (default: a temp dir)')
    ap.add_argument('--tol', type=float, default=1e-4,
                    help='max absolute difference to accept (default 1e-4)')
    args = ap.parse_args()

    import tempfile
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    small = work / 'subset.conllx'
    n = take_sentences(args.conllx, small, args.n_sentences)
    print(f'{n} sentences -> {small}\nworkdir: {work}\n')

    mtype = args.model_type
    if mtype is None:
        from transformers import AutoConfig
        mtype = AutoConfig.from_pretrained(args.model).model_type
    print(f'model_type: {mtype}\n')

    a, b, h = work / 'A.hdf5', work / 'B.hdf5', work / 'H.hdf5'
    print('1. 2+2L extraction (pre-LayerNorm accumulators)')
    run([sys.executable, HERE / 'convert_raw_to_hf_all_checkpoints.py',
         small, a, args.model, '--dtype', 'float32', '--model-dtype', 'float32'])
    print('2. convention B (apply each checkpoint\'s consuming LayerNorm)')
    # Third positional is the HF model id, not the architecture type: the script
    # reloads the model to read its LayerNorm weights.
    run([sys.executable, HERE / 'apply_consuming_layernorm.py', a, b, args.model,
         '--conllx', small, '--validate-sentences', '10'])
    print('3. 1+L extraction (hidden_states)')
    run([sys.executable, HERE / 'convert_raw_to_hf_natural_sentences.py',
         small, h, args.model, '--dtype', 'float32'])

    print('\ncomparing B[2k+1] (post-block, convention B) against H[k+1] '
          '(hidden_states)')
    worst, worst_where, n_cmp = 0.0, None, 0
    with h5py.File(b, 'r') as fb, h5py.File(h, 'r') as fh:
        n_ck_b = fb.attrs.get('n_checkpoints')
        n_ck_h = fh.attrs.get('n_checkpoints')
        keys = sorted(set(fb.keys()) & set(fh.keys()), key=int)
        n_layers = (int(n_ck_b) - 2) // 2
        print(f'  {n_ck_b} checkpoints in B, {n_ck_h} in H -> {n_layers} blocks, '
              f'{len(keys)} sentences')
        for k in keys:
            B, H = np.array(fb[k], dtype=np.float64), np.array(fh[k], dtype=np.float64)
            pairs = [(1, 0)] + [(2 * i + 3, i + 1) for i in range(n_layers)]
            for ib, ih in pairs:
                d = float(np.abs(B[ib] - H[ih]).max())
                n_cmp += 1
                if d > worst:
                    worst, worst_where = d, f'sentence {k}, B[{ib}] vs H[{ih}]'

    print(f'\n  {n_cmp} checkpoint comparisons; max |B - H| = {worst:.3e}'
          f'  (worst at {worst_where})')
    if worst <= args.tol:
        print(f'\nPASS: the 1+L extraction is the post-block subset of the 2+2L '
              f'extraction\n      under convention B, to {args.tol:g}.')
        return 0
    print(f'\nFAIL: difference exceeds {args.tol:g}. Check the consumer-LayerNorm '
          f'map for\n      {mtype} in apply_consuming_layernorm.py.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
