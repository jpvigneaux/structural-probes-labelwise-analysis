'''Extract per-layer hidden states from ANY HuggingFace encoder on natural sentences.

Generalises convert_raw_to_bert_natural_sentences.py to arbitrary encoders
(RoBERTa, DeBERTa-v3, ModernBERT, ...) via AutoModel(output_hidden_states=True).
PTB tokens are rendered to natural English with data.natural_sentence (same as
the BERT extractor), so every model sees the text the way it was pretrained.

Output HDF5 (keys = sentence indices):
  shape (n_saved_layers, seq_len, hidden_dim), where seq_len INCLUDES the model's
  leading/trailing special tokens ([CLS]/[SEP], <s>/</s>, ...). hidden_states[0]
  is the embedding output; hidden_states[k] is the output of transformer block k.
  The pre-aligned probe path (data.BERTDataset, alignment: pre-aligned) drops the
  first/last row and applies the matrix from scripts/precompute_alignments_hf.py,
  so extraction here must use the SAME model+tokenizer as that alignment step.

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
argp.add_argument('--dtype', default='float16', choices=['float16', 'float32'])
args = argp.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
store_dtype = np.float16 if args.dtype == 'float16' else np.float32

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModel.from_pretrained(args.model_name, output_hidden_states=True).to(device).eval()
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
print('Done.')
