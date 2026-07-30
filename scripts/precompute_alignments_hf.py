'''Precompute subword->PTB alignment matrices for ANY HuggingFace encoder.

Generalises precompute_alignments.py (which hardcodes BERT's [CLS]/[SEP] and
guesses the tokenizer from a 'base'/'large' flag) so that every model -- BERT,
RoBERTa, DeBERTa-v3, ModernBERT -- is aligned by exactly the same rule:

  1. de-PTBify the sentence with data.natural_sentence, giving the string the
     model is actually fed (this is our variation on Hewitt & Manning, who feed
     raw PTB tokens);
  2. map subwords to that natural string with the fast tokenizer's exact
     character offsets;
  3. map the natural string back to PTB tokens with a character Levenshtein
     (needed because de-PTBification rewrites characters).

Steps 2-3 are data.hface_alignment_natural, which reproduces the older
BERT-only hface_alignment_deptb exactly (400/400 dev sentences).

The tokenizer is named explicitly with --model-name; nothing is inferred from
hidden_dim, which cannot tell 768-dim models apart.

Output HDF5: key = sentence index, value = (n_subwords_without_specials, n_ptb_words)
float32, columns summing to 1. This is the format consumed by the 'pre-aligned'
branch of data.BERTDataset, which is itself model-agnostic (it just loads the
matrix and does align.T @ features[1:-1]).

Usage:
  python precompute_alignments_hf.py IN.conllx OUT.hdf5 --model-name roberta-base
'''
import os
import sys
from argparse import ArgumentParser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'structural-probes'))

import h5py
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from data import natural_sentence, hface_alignment_natural


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
            buf.append(line.split('\t')[1])
    if buf:
        sentences.append(buf)
    return sentences


def main():
    argp = ArgumentParser()
    argp.add_argument('conllx_path', help='CoNLL-X file (word form = field index 1)')
    argp.add_argument('output_hdf5', help='Output HDF5 for alignment matrices')
    argp.add_argument('--model-name', required=True,
                      help="HuggingFace id whose tokenizer produced the embeddings, "
                           "e.g. bert-base-cased, roberta-base, microsoft/deberta-v3-base")
    args = argp.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if not tokenizer.is_fast:
        raise ValueError(
            f'{args.model_name} has no fast tokenizer; character offsets are required. ')

    sentences = read_conllx_sentences(args.conllx_path)
    print(f'Read {len(sentences)} sentences from {args.conllx_path}')
    print(f'Tokenizer: {args.model_name}')

    with h5py.File(args.output_hdf5, 'w') as fout:
        for idx, tokens in enumerate(tqdm(sentences, desc='[computing alignments]')):
            natural = natural_sentence(tokens)
            enc = tokenizer(natural, return_offsets_mapping=True,
                            return_special_tokens_mask=True)
            mask = enc['special_tokens_mask']

            # Count the leading/trailing specials this tokenizer adds. BERT-family
            # models add one each ([CLS]/[SEP]); GPT-style tokenizers add none, so
            # a hardcoded features[1:-1] would silently drop two real tokens. The
            # counts are stored on the file and honoured when the embeddings are
            # loaded. Specials in the interior are not supported.
            n_pre = 0
            while n_pre < len(mask) and mask[n_pre] == 1:
                n_pre += 1
            n_suf = 0
            while n_suf < len(mask) - n_pre and mask[len(mask) - 1 - n_suf] == 1:
                n_suf += 1
            if sum(mask) != n_pre + n_suf:
                raise ValueError(
                    f'{args.model_name} places special tokens inside the sentence '
                    f'(special_tokens_mask={mask}); this is not supported.')
            if idx == 0:
                first_pre, first_suf = n_pre, n_suf
            elif (n_pre, n_suf) != (first_pre, first_suf):
                raise ValueError(
                    f'inconsistent special-token counts: sentence 0 had '
                    f'({first_pre},{first_suf}), sentence {idx} has ({n_pre},{n_suf}).')

            offsets = [o for o, m in zip(enc['offset_mapping'], mask) if m == 0]
            align = hface_alignment_natural(natural, offsets, tokens)
            fout.create_dataset(str(idx), data=align.numpy().astype(np.float32))

        # Consumed by the pre-aligned branch of data.BERTDataset.
        fout.attrs['n_prefix_special'] = int(first_pre)
        fout.attrs['n_suffix_special'] = int(first_suf)
        fout.attrs['hf_model_name'] = args.model_name
        print(f'special tokens per sentence: {first_pre} leading, {first_suf} trailing')

    print('Done.')


if __name__ == '__main__':
    main()
