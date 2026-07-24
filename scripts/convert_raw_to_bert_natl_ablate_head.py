'''
Extract BERT hidden states from PTB tokens converted to natural English while
zero-ablating one attention head with TransformerLens.

This mirrors convert_raw_to_bert_natural_sentences.py:
  - Input: CoNLL-X file; word form = field index 1.
  - Output: HDF5 with keys = sentence indices.
  - Each dataset has shape (CHECKPOINT_COUNT_TOTAL, seq_len, hidden_dim).
  - seq_len includes [CLS] and [SEP].

Additional behavior:
  - The requested attention head is set to zero at blocks.{layer}.attn.hook_z.
  - Layer and head indices are 0-indexed, matching TransformerLens hook names.

HDF5 index scheme (CHECKPOINT_COUNT_TOTAL = 26 for BERT-base):
  0          emb       raw word embeddings (no position, no LayerNorm)
  1          layer-00  embedding output (word + position + type + LayerNorm)
  2k         layer-k+A post-attention intermediate of block k, k=1..12
             = LayerNorm(layer_{k-1} + MultiHeadAttn(layer_{k-1}))
  2k+1       layer-k   full output of transformer block k, k=1..12
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
argp.add_argument('input_file', help='CoNLL-X (.conllx) file; word form is field index 1')
argp.add_argument('output_dir', help='Output HDF5 dir (not file name)')
argp.add_argument('bert_model', choices=['base', 'large'], help='base or large')
argp.add_argument('--device', default=None,
                  help='Optional torch device, e.g. cuda or cpu. Defaults to cuda if available.')
argp.add_argument('--layers', type=int, nargs='+', default=None, metavar='L',
                  help='Attention layers to ablate (0-indexed). Defaults to all layers.')
args = argp.parse_args()

if args.bert_model == 'base':
    model_name = 'bert-base-cased'
    LAYER_COUNT = 12
    FEATURE_COUNT = 768
    N_HEADS = 12
elif args.bert_model == 'large':
    model_name = 'bert-large-cased'
    LAYER_COUNT = 24
    FEATURE_COUNT = 1024
    N_HEADS = 16
else:
    raise ValueError("bert_model must be base or large")

CHECKPOINT_COUNT_TOTAL = 2 + LAYER_COUNT * 2  # emb + layer-00 + 2 per block

device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = HookedEncoder.from_pretrained(model_name)
bert.to(device)
bert.eval()


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




def make_zero_head_hook(head):
    def zero_selected_head(z, hook):
        """Zero one attention head at hook_z: [batch, pos, head, d_head]."""
        z[:, :, head, :] = 0.0
        return z
    return zero_selected_head

sentences = read_conllx_sentences(args.input_file)

import re as _re
_stem  = os.path.splitext(os.path.basename(args.input_file))[0]
_match = _re.search(r'\b(train|dev|test)\b', _stem)
split  = _match.group(1) if _match else _stem  # e.g. "dev", "train", "test"

os.makedirs(args.output_dir, exist_ok=True)

layers_to_run = args.layers if args.layers is not None else list(range(LAYER_COUNT))

for layer in layers_to_run:
    for head in range(N_HEADS):
        print(f"Working on Layer {layer} Head {head}")
        output_filepath = f"{args.output_dir}/raw.{split}.L{layer}H{head}ablated.bertbase-natural-layers.hdf5"


        ablation_hook_name = f'blocks.{layer}.attn.hook_z' ##this specifies the layer to ablate

        z_head_hook = make_zero_head_hook(head) ##this specifies the head to ablate
        with h5py.File(output_filepath, 'w') as fout:


            for index, tokens in enumerate(sentences):
                natural = natural_sentence(tokens)

                tokenized = ['[CLS]'] + tokenizer.tokenize(natural) + ['[SEP]']
                indexed = tokenizer.convert_tokens_to_ids(tokenized)
                seg_ids = [0] * len(tokenized)

                input_ids = torch.tensor([indexed], device=device)
                token_type_ids = torch.tensor([seg_ids], device=device)

                with torch.no_grad():
                    with bert.hooks(fwd_hooks=[(ablation_hook_name, z_head_hook)]):
                        _, cache = bert.run_with_cache(
                            input_ids,
                            token_type_ids=token_type_ids,
                        )

                seq_len = len(tokenized)
                dset = fout.create_dataset(
                    str(index), (CHECKPOINT_COUNT_TOTAL, seq_len, FEATURE_COUNT), dtype=np.float32
                )

                # Index 0: raw word token embeddings (no position, no LayerNorm)
                dset[0] = cache['embed.hook_embed'][0].detach().cpu().numpy()

                # Index 1: full embedding output (word + position + type + LayerNorm)
                dset[1] = cache['blocks.0.hook_resid_pre'][0].detach().cpu().numpy()

                for k in range(LAYER_COUNT):
                    dset[2 + 2 * k] = cache[f'blocks.{k}.hook_resid_mid'][0].detach().cpu().numpy()
                    dset[2 + 2 * k + 1] = cache[f'blocks.{k}.hook_resid_post'][0].detach().cpu().numpy()

                if (index + 1) % 500 == 0:
                    print(f'Processed {index + 1} sentences')

        print('Done.')
