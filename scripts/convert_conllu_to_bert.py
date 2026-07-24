'''
Extract BERT-base hidden states from CoNLL-U sentences.

UD tokens are already surface forms; spacing is reconstructed with
natural_sentence_ud() using the SpaceAfter=No MISC annotation.
Sentences that are not natural language (URLs, email addresses, too
short, no alphabetic content) are filtered with is_natural_sentence_ud()
before extraction.

A filtered CoNLL-U file is written alongside the HDF5 so that HDF5
key i corresponds exactly to sentence i in the filtered file.

HDF5 index scheme (identical to convert_raw_to_bert_natural_sentences.py):
  0          raw word embeddings
  1          layer-00  (embedding output)
  2k         layer-k post-attention intermediate, k=1..12
  2k+1       layer-k full block output,           k=1..12

Usage:
  python convert_conllu_to_bert.py <input.conllu> <output.hdf5> \
      --filtered-conllu <filtered.conllu> [--bert-model base|large]
'''
import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structural-probes'))

import torch
import h5py
import numpy as np
from argparse import ArgumentParser
from transformer_lens import HookedEncoder
from transformers import AutoTokenizer

from data import natural_sentence_ud, is_natural_sentence_ud

argp = ArgumentParser()
argp.add_argument('input_conllu',    help='CoNLL-U input file')
argp.add_argument('output_hdf5',     help='Output HDF5 file')
argp.add_argument('--filtered-conllu', required=True,
                  help='Write filtered CoNLL-U sentences here (HDF5 key i = sentence i)')
argp.add_argument('--bert-model', default='base', choices=['base', 'large'])
args = argp.parse_args()

if args.bert_model == 'base':
    model_name = 'bert-base-cased'
    LAYER_COUNT = 12
    FEATURE_COUNT = 768
else:
    model_name = 'bert-large-cased'
    LAYER_COUNT = 24
    FEATURE_COUNT = 1024

LAYER_COUNT_TOTAL = 2 + LAYER_COUNT * 2

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = HookedEncoder.from_pretrained(model_name)
bert.eval()


def iter_conllu(path):
    '''Yield (token_rows, raw_lines) per sentence, skipping comment and MWT lines.'''
    buf_rows, buf_lines = [], []
    with open(path) as fh:
        for line in fh:
            raw = line.rstrip('\n')
            if raw.startswith('#'):
                buf_lines.append(raw)
                continue
            if re.match(r'^\d+[-\.]\d+', raw):
                continue
            if not raw.strip():
                if buf_rows:
                    yield buf_rows, buf_lines
                    buf_rows, buf_lines = [], []
                else:
                    buf_lines = []
                continue
            fields = raw.split('\t')
            buf_rows.append(fields)
            buf_lines.append(raw)
    if buf_rows:
        yield buf_rows, buf_lines


total_read = kept = 0
with h5py.File(args.output_hdf5, 'w') as fout, \
     open(args.filtered_conllu, 'w') as fconllu:

    for rows, lines in iter_conllu(args.input_conllu):
        total_read += 1
        tokens = [r[1] for r in rows]
        misc   = [r[9] for r in rows]

        if not is_natural_sentence_ud(tokens):
            continue

        # Write sentence to filtered CoNLL-U
        for line in lines:
            fconllu.write(line + '\n')
        fconllu.write('\n')

        natural   = natural_sentence_ud(tokens, misc)
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
        dset = fout.create_dataset(str(kept), (LAYER_COUNT_TOTAL, seq_len, FEATURE_COUNT))

        dset[0] = cache['embed.hook_embed'][0].cpu().numpy()
        dset[1] = cache['blocks.0.hook_resid_pre'][0].cpu().numpy()
        for k in range(LAYER_COUNT):
            dset[2 + 2 * k]     = cache[f'blocks.{k}.hook_resid_mid'][0].cpu().numpy()
            dset[2 + 2 * k + 1] = cache[f'blocks.{k}.hook_resid_post'][0].cpu().numpy()

        kept += 1
        if kept % 500 == 0:
            print(f'  {kept} sentences processed')

print(f'Done. Read {total_read}, kept {kept} (dropped {total_read - kept}).')
