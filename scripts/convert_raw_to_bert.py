'''
Takes raw PTB-tokenised text and saves BERT hidden states to disk.

Input: one space-separated PTB sentence per line.
Output: HDF5 with keys = sentence indices,
        shape (LAYER_COUNT, seq_len, hidden_dim)  — one row per transformer block.
        seq_len includes [CLS] and [SEP].
'''
import torch
from transformer_lens import HookedEncoder
from transformers import AutoTokenizer
from argparse import ArgumentParser
import h5py
import numpy as np

argp = ArgumentParser()
argp.add_argument('input_path')
argp.add_argument('output_path')
argp.add_argument('bert_model', help='base or large')
args = argp.parse_args()

if args.bert_model == 'base':
    model_name   = 'bert-base-cased'
    LAYER_COUNT  = 12
    FEATURE_COUNT = 768
elif args.bert_model == 'large':
    model_name   = 'bert-large-cased'
    LAYER_COUNT  = 24
    FEATURE_COUNT = 1024
else:
    raise ValueError("BERT model must be base or large")

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert = HookedEncoder.from_pretrained(model_name)
bert.eval()

with h5py.File(args.output_path, 'w') as fout:
    for index, line in enumerate(open(args.input_path)):
        line = line.strip()
        tokenized_text = ['[CLS]'] + tokenizer.tokenize(line) + ['[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        seg_ids        = [0] * len(tokenized_text)

        input_ids      = torch.tensor([indexed_tokens])
        token_type_ids = torch.tensor([seg_ids])

        with torch.no_grad():
            _, cache = bert.run_with_cache(
                input_ids,
                token_type_ids=token_type_ids,
            )

        seq_len = len(tokenized_text)
        dset = fout.create_dataset(str(index), (LAYER_COUNT, seq_len, FEATURE_COUNT))

        for k in range(LAYER_COUNT):
            dset[k] = cache[f'blocks.{k}.hook_resid_post'][0].cpu().numpy()
