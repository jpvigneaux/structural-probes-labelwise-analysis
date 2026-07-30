'''Extract post-block hidden states from a post-LayerNorm HuggingFace encoder.

This is the `1+L` extractor: one checkpoint per transformer block, plus the
embedding output, taken straight from `output_hidden_states=True`. It produced
the DeBERTa-v3-base and RoBERTa-Shuffle-n1 runs in the paper, whose checkpoints
are therefore numbered 0..L rather than 0..2L+1 -- see experiments/paper_runs.yaml.
For the `2+2L` layout, which additionally recovers the mid-block (post-attention)
residual stream, use convert_raw_to_hf_all_checkpoints.py.

Restricted to post-LayerNorm architectures (BERT, RoBERTa, DeBERTa-v2/v3),
because for a pre-LayerNorm model HuggingFace applies the model's FINAL norm
(`ln_f`, `final_norm`) to hidden_states[-1] and to no other entry, which would
silently put the last checkpoint in a different space from its neighbours. Pass
--allow-pre-ln if you want that anyway; convert_raw_to_hf_all_checkpoints.py
avoids the problem by reading block outputs directly.

For a post-LN block, hidden_states[k] is the post-LayerNorm residual stream --
Hewitt & Manning's convention, and "convention B" in the sense of
apply_consuming_layernorm.py. It needs no LayerNorm conversion afterwards.

Output HDF5 (keys = sentence indices):
  shape (n_saved_layers, seq_len, hidden_dim), where seq_len INCLUDES the model's
  leading/trailing special tokens ([CLS]/[SEP], <s>/</s>, ...). hidden_states[0]
  is the embedding output; hidden_states[k] is the output of transformer block k.
  The pre-aligned probe path (data.BERTDataset, alignment: pre-aligned) strips
  the special-token rows named in the alignment file and applies the matrix from
  scripts/precompute_alignments_hf.py, so extraction here must use the SAME
  model+tokenizer as that alignment step.

Usage:
  python convert_raw_to_hf_natural_sentences.py INPUT.conllx OUT.hdf5 MODEL_NAME \
      [--layers all|0,3,5,7] [--dtype float16|float32]
'''
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structural-probes'))

import h5py
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from data import natural_sentence

argp = ArgumentParser()
argp.add_argument('input_path', help='CoNLL-X file; word form = field index 1')
argp.add_argument('output_path', help='Output HDF5 file')
argp.add_argument('model_name', help='HF model id, e.g. roberta-base, microsoft/deberta-v3-base')
argp.add_argument('--layers', default='all',
                  help='"all" or comma-separated hidden_state indices (0=embeddings, 1..L=blocks)')
argp.add_argument('--dtype', default='float16', choices=['float16', 'float32'],
                  help='dtype the hidden states are STORED in')
argp.add_argument('--revision', default=None,
                  help='HF revision/branch to load, e.g. float16 for EleutherAI/gpt-j-6B. '
                       'Prefer a released low-precision branch over casting the fp32 main '
                       'branch yourself: for GPT-J the two differ by one ulp in wte.weight.')
argp.add_argument('--model-dtype', default='float32', choices=['float16', 'float32'],
                  help='dtype the model WEIGHTS are loaded in. float16 halves both host\n'
                       'and device memory; use it for multi-billion-parameter models such\n'
                       'as GPT-J, whose fp32 checkpoint is 23GB.')
argp.add_argument('--allow-pre-ln', action='store_true',
                  help='extract a pre-LayerNorm model anyway, accepting that '
                       'hidden_states[-1] carries the final norm and the others do not')
args = argp.parse_args()

# Post-LayerNorm architectures, for which every hidden_states entry is the
# post-LayerNorm residual stream and the series is internally consistent.
POST_LN_TYPES = {'bert', 'roberta', 'deberta', 'deberta-v2', 'electra', 'xlm-roberta'}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
store_dtype = np.float16 if args.dtype == 'float16' else np.float32

tokenizer = AutoTokenizer.from_pretrained(args.model_name, revision=args.revision)
import torch as _torch
_wdtype = _torch.float16 if args.model_dtype == 'float16' else _torch.float32
# low_cpu_mem_usage avoids materialising the full fp32 checkpoint in host RAM
# before casting, which for GPT-J would spike to 23GB.
model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True,
                                  dtype=_wdtype, low_cpu_mem_usage=True,
                                  revision=args.revision).to(device).eval()
mtype = model.config.model_type
if mtype not in POST_LN_TYPES and not args.allow_pre_ln:
    raise SystemExit(
        f"model_type '{mtype}' is not a known post-LayerNorm encoder. HuggingFace "
        f"applies the model's final norm to hidden_states[-1] only, so for a "
        f"pre-LayerNorm model the last checkpoint would not live in the same space "
        f"as the rest. Use convert_raw_to_hf_all_checkpoints.py, which reads block "
        f"outputs directly, or pass --allow-pre-ln to override.")

n_layers = model.config.num_hidden_layers
hidden = model.config.hidden_size
n_states = n_layers + 1  # embeddings + one per block
if args.layers == 'all':
    keep = list(range(n_states))
else:
    keep = [int(x) for x in args.layers.split(',')]
    assert all(0 <= k < n_states for k in keep), f'layers must be in [0,{n_states-1}]'
print(f'{args.model_name}: {n_layers} layers, hidden={hidden}, saving hidden_states {keep}, dtype={args.dtype}')


def read_conllx_sentences(path):
    sentences, buf = [], []
    for line in open(path):
        line = line.strip()
        if line.startswith('#'):
            continue
        if not line:
            if buf:
                sentences.append(buf); buf = []
        else:
            buf.append(line.split('\t')[1])
    if buf:
        sentences.append(buf)
    return sentences


sentences = read_conllx_sentences(args.input_path)
print(f'Read {len(sentences)} sentences from {args.input_path}')

with h5py.File(args.output_path, 'w') as fout:
    for index, tokens in enumerate(sentences):
        natural = natural_sentence(tokens)
        enc = tokenizer(natural, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model(**enc)
        # hidden_states: tuple(n_states) each (1, seq_len, hidden)
        hs = torch.stack([out.hidden_states[k][0] for k in keep], dim=0)  # (n_keep, seq_len, hidden)
        dset = fout.create_dataset(str(index), data=hs.cpu().numpy().astype(store_dtype))
        if (index + 1) % 500 == 0:
            print(f'Processed {index + 1} sentences')

    # Same provenance attributes as convert_raw_to_hf_all_checkpoints.py, so a
    # stored file says which layout it is in rather than leaving it to be
    # inferred from the checkpoint count. scripts/check_embeddings.py reads these.
    fout.attrs['n_checkpoints'] = len(keep)
    fout.attrs['has_midblock'] = False
    fout.attrs['hf_model_name'] = args.model_name
    fout.attrs['layout'] = '0=embed_out,k=post_block'
    fout.attrs['convention'] = 'B:consumed'
    fout.attrs['convention_note'] = ('post-LayerNorm hidden_states; for these '
                                     'architectures conventions A and B coincide')
print('Done.')
