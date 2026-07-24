'''
Extract BERT-base hidden states from PTB tokens converted to natural English.

PTB escapes (-LRB-, -RRB-, ``, '', etc.) are replaced with their surface forms
and joined with natural spacing before being fed to BERT, so representations
reflect how the model was actually pretrained rather than PTB-specific markup.
PTB spacing logic lives in data.natural_sentence; this script imports it so
there is a single source of truth between extraction and alignment.

HDF5 index scheme (LAYER_COUNT_TOTAL = 26 for BERT-base):
  0          emb       raw word embeddings (no position, no LayerNorm)
  1          layer-00  embedding output (word + position + type + LayerNorm)
  2k         layer-k+A post-attention intermediate of block k, k=1..12
             = LayerNorm(layer_{k-1} + MultiHeadAttn(layer_{k-1}))
  2k+1       layer-k   full output of transformer block k, k=1..12

Explicit mapping for BERT-base:
  idx  name
   0   emb
   1   layer-00
   2   layer-01+A   4  layer-02+A   6  layer-03+A   8  layer-04+A
   3   layer-01     5  layer-02     7  layer-03     9  layer-04
  10   layer-05+A  12  layer-06+A  14  layer-07+A  16  layer-08+A
  11   layer-05    13  layer-06    15  layer-07    17  layer-08
  18   layer-09+A  20  layer-10+A  22  layer-11+A  24  layer-12+A
  19   layer-09    21  layer-10    23  layer-11    25   layer-12

Input: CoNLL-X file; word form = field index 1.
Output: HDF5 with keys = sentence indices, shape (LAYER_COUNT_TOTAL, seq_len, hidden_dim).
        seq_len includes [CLS] and [SEP].
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structural-probes'))

import torch
import h5py
import numpy as np
from argparse import ArgumentParser
from transformer_lens import HookedEncoder
from transformers import AutoTokenizer

from data import natural_sentence

argp = ArgumentParser()
argp.add_argument('input_path', help='CoNLL-X (.conllx) file; word form is field index 1')
argp.add_argument('output_path', help='Output HDF5 file')
argp.add_argument('bert_model', help='base or large')
args = argp.parse_args()

if args.bert_model == 'base':
    model_name = 'bert-base-cased'
    LAYER_COUNT = 12
    FEATURE_COUNT = 768
elif args.bert_model == 'large':
    model_name = 'bert-large-cased'
    LAYER_COUNT = 24
    FEATURE_COUNT = 1024
else:
    raise ValueError("bert_model must be base or large")

LAYER_COUNT_TOTAL = 2 + LAYER_COUNT * 2  # emb + layer-00 + 2 per block

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = HookedEncoder.from_pretrained(model_name)
bert.eval()

# ---------------------------------------------------------------------------
# Parse CoNLL-X into sentences (word form = field index 1)
# ---------------------------------------------------------------------------
def read_conllx_sentences(path):
    sentences = []
    buf = []
    for line in open(path):
        line = line.strip()
        if line.startswith('#'):
            continue
        if not line:
            if buf:
                sentences.append(buf)
                buf = []
        else:
            fields = line.split('\t')
            buf.append(fields[1])
    if buf:
        sentences.append(buf)
    return sentences

sentences = read_conllx_sentences(args.input_path)
print(f'Read {len(sentences)} sentences from {args.input_path}')

# ---------------------------------------------------------------------------
# Print all natural sentences that will be fed to BERT
# ---------------------------------------------------------------------------
print('\n--- Natural sentences fed to BERT ---')
for idx, tokens in enumerate(sentences):
    nat = natural_sentence(tokens)
    print(f'{idx}\t{nat}')
print('--- End of sentence list ---\n')

# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------
with h5py.File(args.output_path, 'w') as fout:
    for index, tokens in enumerate(sentences):
        natural = natural_sentence(tokens)

        # Tokenise: ['[CLS]', ...subwords..., '[SEP]']
        tokenized = ['[CLS]'] + tokenizer.tokenize(natural) + ['[SEP]']
        indexed   = tokenizer.convert_tokens_to_ids(tokenized)
        seg_ids   = [0] * len(tokenized)

        input_ids      = torch.tensor([indexed])
        token_type_ids = torch.tensor([seg_ids])

        with torch.no_grad():
            _, cache = bert.run_with_cache(
                input_ids,
                token_type_ids=token_type_ids,
            )

        seq_len = len(tokenized)
        dset = fout.create_dataset(
            str(index), (LAYER_COUNT_TOTAL, seq_len, FEATURE_COUNT)
        )

        # Index 0: raw word token embeddings (no position, no LayerNorm)
        dset[0] = cache['embed.hook_embed'][0].cpu().numpy()

        # Index 1: full embedding output (word + position + type + LayerNorm)
        dset[1] = cache['blocks.0.hook_resid_pre'][0].cpu().numpy()

        for k in range(LAYER_COUNT):
            # post-attention intermediate: LayerNorm(attn_out + residual)
            dset[2 + 2 * k]     = cache[f'blocks.{k}.hook_resid_mid'][0].cpu().numpy()
            # full block output (attention + FFN)
            dset[2 + 2 * k + 1] = cache[f'blocks.{k}.hook_resid_post'][0].cpu().numpy()

        if (index + 1) % 500 == 0:
            print(f'Processed {index + 1} sentences')

print('Done.')
