'''
Pre-compute hface-deptb alignment matrices for BERT natural-sentence HDF5 files.

Alignment is a property of (PTB tokens, tokeniser) only — it does not depend on
which layer is being probed.  Saving the matrices once lets run_experiment.py
skip the Levenshtein step entirely, replacing it with a cheap matmul per layer.

Output HDF5 layout (one file per corpus split):
  key = str(sentence_index)
  value = align_mat of shape (n_subwords_no_cls_sep, n_ptb_words), float32
          columns sum to 1 (column-normalised)

Usage:
  python precompute_alignments.py <conllx> <output_hdf5> base|large
'''

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structural-probes'))

import h5py
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer

from data import natural_sentence, hface_alignment_deptb


def read_conllx_sentences(path):
    sentences, buf = [], []
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


argp = ArgumentParser()
argp.add_argument('conllx_path', help='CoNLL-X file (word form = field index 1)')
argp.add_argument('output_hdf5', help='Output HDF5 for alignment matrices')
argp.add_argument('bert_model',  help='base or large')
args = argp.parse_args()

if args.bert_model == 'base':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
elif args.bert_model == 'large':
    tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
else:
    raise ValueError("bert_model must be 'base' or 'large'")

sentences = read_conllx_sentences(args.conllx_path)
print(f'Read {len(sentences)} sentences from {args.conllx_path}')

with h5py.File(args.output_hdf5, 'w') as fout:
    for idx, tokens in enumerate(tqdm(sentences, desc='[computing alignments]')):
        natural  = natural_sentence(tokens)
        tok_sent = ['[CLS]'] + tokenizer.tokenize(natural) + ['[SEP]']
        align    = hface_alignment_deptb(tok_sent, tokens).numpy().astype(np.float32)
        fout.create_dataset(str(idx), data=align)

print('Done.')
