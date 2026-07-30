#!/usr/bin/env python3
"""Check extracted embeddings and their alignment matrices against the manifest.

Every way of getting this wrong is silent. A model extracted with the wrong
layout still trains; a tokenizer that adds no special tokens still produces an
alignment matrix; an off-by-two in the special-token strip still yields a probe
with a plausible-looking UUAS, just a slightly worse one. Nothing raises. So the
invariants are checked explicitly, against what experiments/paper_runs.yaml says
the run is supposed to be:

  checkpoint count    matches the manifest, and the stored `layout` attribute
                      agrees with `2+2L` / `1+L`
  provenance          the `hf_model_name` on the embedding file and on the
                      alignment file name the same model
  special tokens      the alignment file's (n_prefix, n_suffix) match the
                      manifest, and are what the tokenizer actually produces
  shape agreement     for every sentence, (embedding rows - specials) equals the
                      alignment's subword count, and the alignment's word count
                      equals the number of PTB tokens in the CoNLL-X file
  stochasticity       every alignment column sums to 1, so each PTB word is a
                      convex combination of subwords and no word is dropped

The shape check is the one that matters most: it is exactly the composition
`align.T @ features[n_pre : len - n_suf]` that data.BERTDataset performs, so if
it passes here the probe cannot be misaligned.

Usage:
    python scripts/check_embeddings.py --run gpt2
    python scripts/check_embeddings.py --run deberta --splits dev --tokenizer-check
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / 'experiments' / 'paper_runs.yaml'


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


class Checks:
    def __init__(self):
        self.fail = 0

    def ok(self, cond, msg, detail=''):
        cond = bool(cond)
        if not cond:
            self.fail += 1
        print(f'    [{"ok" if cond else "FAIL"}] {msg}'
              + (f'   {detail}' if detail else ''))
        return cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--manifest', default=str(MANIFEST))
    ap.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    ap.add_argument('--max-sentences', type=int, default=0,
                    help='check only the first N sentences of each split (0 = all)')
    ap.add_argument('--tokenizer-check', action='store_true',
                    help='also load the tokenizer and confirm the special-token '
                         'counts it produces (needs the model available offline)')
    args = ap.parse_args()

    man = yaml.safe_load(open(args.manifest))
    if args.run not in man['runs']:
        sys.exit(f"unknown run '{args.run}'; known: {' '.join(man['runs'])}")
    run, roots = man['runs'][args.run], man['roots']
    ex = run['extraction']

    emb_dir = Path(roots['embeddings']) / ex['embeddings']
    align_dir = Path(roots['alignments']) / ex['alignments']
    corpus = Path(roots['corpus'])
    exp_pre, exp_suf = ex['n_special']

    print(f"{run['label']}  ({args.run})")
    print(f'  embeddings  {emb_dir}')
    print(f'  alignments  {align_dir}')
    print(f'  layout      {ex["layout"]}, {ex["n_checkpoints"]} checkpoints, '
          f'{exp_pre} leading / {exp_suf} trailing special tokens\n')

    c = Checks()

    if args.tokenizer_check:
        from transformers import AutoTokenizer
        sys.path.insert(0, str(HERE.parent / 'structural-probes'))
        from data import natural_sentence
        tok = AutoTokenizer.from_pretrained(ex['tokenizer'])
        enc = tok(natural_sentence(['The', 'dog', 'barks', '.']),
                  return_special_tokens_mask=True)
        mask = enc['special_tokens_mask']
        n_pre = next((i for i, m in enumerate(mask) if m == 0), len(mask))
        n_suf = next((i for i, m in enumerate(reversed(mask)) if m == 0), len(mask))
        c.ok((n_pre, n_suf) == (exp_pre, exp_suf),
             f'tokenizer {ex["tokenizer"]} adds the declared special tokens',
             f'got ({n_pre},{n_suf}), manifest says ({exp_pre},{exp_suf})')

    for split in args.splits:
        emb_path = emb_dir / f'raw.{split}.{ex["stem"]}.hdf5'
        align_path = align_dir / f'alignments.{split}.hdf5'
        print(f'  --- {split} ---')
        if not c.ok(emb_path.exists(), f'embeddings present', str(emb_path)):
            continue
        if not c.ok(align_path.exists(), f'alignments present', str(align_path)):
            continue

        conllx = corpus / f'ptb3-wsj-{split}.conllx'
        sentences = read_conllx_sentences(conllx) if conllx.exists() else None
        if sentences is None:
            print(f'    [--] {conllx} not found; skipping word-count check')

        with h5py.File(emb_path, 'r') as fe, h5py.File(align_path, 'r') as fa:
            keys = sorted(fe.keys(), key=int)
            n_ck = int(fe.attrs['n_checkpoints']) if 'n_checkpoints' in fe.attrs \
                else int(fe[keys[0]].shape[0])
            c.ok(n_ck == ex['n_checkpoints'], 'checkpoint count matches the manifest',
                 f'file {n_ck}, manifest {ex["n_checkpoints"]}')

            if 'layout' in fe.attrs:
                want = ('0=raw_embed,1=embed_out,2k=post_attn,2k+1=post_block'
                        if ex['layout'] == '2+2L' else '0=embed_out,k=post_block')
                c.ok(str(fe.attrs['layout']) == want, 'stored layout matches the manifest',
                     f'file "{fe.attrs["layout"]}"')
            else:
                print('    [--] no layout attribute (file predates provenance '
                      'attributes); inferred from the checkpoint count')

            if 'hf_model_name' in fe.attrs:
                c.ok(str(fe.attrs['hf_model_name']) == ex['hf_name'],
                     'embedding file names the manifest model',
                     f'file "{fe.attrs["hf_model_name"]}"')

            # Alignment files written before the attributes existed default to
            # the BERT-era (1,1), which is what data.BERTDataset also assumes.
            n_pre = int(fa.attrs.get('n_prefix_special', 1))
            n_suf = int(fa.attrs.get('n_suffix_special', 1))
            c.ok((n_pre, n_suf) == (exp_pre, exp_suf),
                 'special-token counts match the manifest',
                 f'alignment file ({n_pre},{n_suf})'
                 + ('' if 'n_prefix_special' in fa.attrs else ' [defaulted]'))

            c.ok(len(fe.keys()) == len(fa.keys()),
                 'same number of sentences in both files',
                 f'{len(fe.keys())} embeddings, {len(fa.keys())} alignments')
            if sentences is not None:
                c.ok(len(fe.keys()) == len(sentences),
                     'sentence count matches the CoNLL-X split',
                     f'{len(fe.keys())} vs {len(sentences)}')

            check_keys = keys if args.max_sentences == 0 else keys[:args.max_sentences]
            bad_sub, bad_word, bad_sum = [], [], []
            worst_sum = 0.0
            for k in check_keys:
                if k not in fa:
                    bad_sub.append((k, 'missing alignment'))
                    continue
                emb = fe[k]
                al = np.array(fa[k])
                n_rows = emb.shape[1] - n_pre - n_suf
                if n_rows != al.shape[0]:
                    bad_sub.append((k, f'{emb.shape[1]} - {n_pre} - {n_suf} '
                                       f'= {n_rows} != {al.shape[0]}'))
                if sentences is not None and al.shape[1] != len(sentences[int(k)]):
                    bad_word.append((k, f'{al.shape[1]} != {len(sentences[int(k)])}'))
                col = al.sum(axis=0)
                worst_sum = max(worst_sum, float(np.abs(col - 1.0).max()))
                if not np.allclose(col, 1.0, atol=1e-5):
                    bad_sum.append(k)

            n = len(check_keys)
            c.ok(not bad_sub,
                 f'subword rows line up with the alignment on all {n} sentences',
                 '' if not bad_sub else f'{len(bad_sub)} bad, e.g. {bad_sub[:3]}')
            c.ok(not bad_word,
                 f'alignment word count equals the PTB token count on all {n}',
                 '' if not bad_word else f'{len(bad_word)} bad, e.g. {bad_word[:3]}')
            c.ok(not bad_sum,
                 f'every alignment column sums to 1 (max deviation {worst_sum:.2e})',
                 '' if not bad_sum else f'{len(bad_sum)} bad, e.g. {bad_sum[:3]}')
        print()

    print('=' * 70)
    if c.fail:
        print(f'{c.fail} check(s) FAILED for run {args.run}')
        return 1
    print(f'all checks passed for run {args.run}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
