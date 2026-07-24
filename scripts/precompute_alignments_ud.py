'''
Pre-compute hface-ud alignment matrices for BERT CoNLL-U HDF5 files.

UD analogue of precompute_alignments.py.  Reads a *filtered* CoNLL-U file
(produced by convert_conllu_to_bert.py) so HDF5 key i == sentence i.

Usage:
  python precompute_alignments_ud.py <filtered.conllu> <output_alignments.hdf5> base|large
'''
import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structural-probes'))

import h5py
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer

from data import natural_sentence_ud, hface_alignment_ud


def iter_conllu(path):
    buf = []
    with open(path) as fh:
        for line in fh:
            raw = line.rstrip('\n')
            if raw.startswith('#') or re.match(r'^\d+[-\.]\d+', raw):
                continue
            if not raw.strip():
                if buf:
                    yield buf
                    buf = []
                continue
            buf.append(raw.split('\t'))
    if buf:
        yield buf


argp = ArgumentParser()
argp.add_argument('conllu_path',  help='Filtered CoNLL-U file')
argp.add_argument('output_hdf5',  help='Output HDF5 for alignment matrices')
argp.add_argument('bert_model',   help='base or large')
args = argp.parse_args()

if args.bert_model == 'base':
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
elif args.bert_model == 'large':
    tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
else:
    raise ValueError("bert_model must be 'base' or 'large'")

sentences = list(iter_conllu(args.conllu_path))
print(f'Read {len(sentences)} sentences from {args.conllu_path}')

with h5py.File(args.output_hdf5, 'w') as fout:
    for idx, rows in enumerate(tqdm(sentences, desc='[computing alignments]')):
        tokens  = [r[1] for r in rows]
        misc    = [r[9] for r in rows]
        natural = natural_sentence_ud(tokens, misc)
        tok_sent = ['[CLS]'] + tokenizer.tokenize(natural) + ['[SEP]']
        align    = hface_alignment_ud(tok_sent, tokens, misc).numpy().astype(np.float32)
        fout.create_dataset(str(idx), data=align)

print('Done.')
