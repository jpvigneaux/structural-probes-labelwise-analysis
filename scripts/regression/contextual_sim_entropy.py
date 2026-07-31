#!/usr/bin/env python3
"""Similarity-corrected entropy per dependency relation, from EITHER
model-internal contextualised embeddings OR static fastText vectors.

Motivation
----------
The published analysis measured lexical similarity Z(x,y) in an external static
space (fastText). Measuring it instead in the *probed model's own* space asks a
more direct question -- whether the words filling a relation's head slot are
spread out or collapsed together in exactly the representation the structural
probe reads -- and it makes the diversity measure model-specific, so it can be
recomputed for every model in the comparison.

Both are supported here so the two can be reported side by side. They answer
different questions and neither strictly dominates:

  --vec        similarity is INDEPENDENT of the probed model. The predictor is
               external, so "low diversity predicts high ULAS" is a genuinely
               out-of-sample claim.
  --embeddings similarity comes from the same representation the probe reads.
               More direct, but the predictor and the outcome now share a
               geometry, so the result is a weaker form of evidence.

Word types, not tokens
----------------------
The entropy is defined over word *types* (p_r(x) is the probability of type x
filling a role of relation r), whereas contextual vectors are per token. Each
type is therefore represented by the mean of its contextual vectors over all
corpus occurrences -- the standard "static-ised" contextual embedding of
Bommasani et al. (2020). Word-level vectors are obtained with the same
alignment matmul the probe uses (`align.T @ subword_features`), so the space is
exactly the probe's input space.

Entropy
-------
Matches the published definition:

    q(x)      = sum_y max(cos(x,y), 0) * p(y)        [diagonal term is p(x)]
    H_sim(r)  = -sum_x p(x) * log2 q(x)

q is computed in row chunks of a dense per-relation cosine matrix. The sparse
earlier global-matrix approach needed one entry per
within-relation word pair, which on PTB-train is 756 million pairs (~23 GB) and
OOMs; chunking never materialises more than `--chunk` rows at a time.

Usage
-----
  # model-internal (BERT-base checkpoint 16)
  python contextual_sim_entropy.py PTB.conllx --out sim_bertbase_ck16.tsv \\
      --embeddings raw.train.bertbase-natural-layers.hdf5 \\
      --alignments alignments.train.hdf5 --layer 16 --pca-dim 50

  # static fastText, same entropy code path
  python contextual_sim_entropy.py PTB.conllx --out sim_fasttext.tsv \\
      --vec wiki-news-300d-1M.vec --pca-dim 50
"""

import argparse
import math
import sys
from collections import defaultdict

import numpy as np


def parse_conllx(path):
    """Yield sentences as lists of (form, head_1indexed, deprel)."""
    sent = []
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        if line.startswith('#'):
            continue
        if not line:
            if sent:
                yield sent
                sent = []
            continue
        f = line.split('\t')
        if len(f) < 8 or '-' in f[0]:
            continue
        sent.append((f[1], int(f[6]), f[7]))
    if sent:
        yield sent


def relation_vocabularies(sentences, lowercase=False):
    """counts[relation][role][word_type] -> occurrence count."""
    counts = {'head': defaultdict(lambda: defaultdict(int)),
              'dep': defaultdict(lambda: defaultdict(int))}
    for sent in sentences:
        for i, (form, head, rel) in enumerate(sent):
            if head == 0 or head > len(sent):
                continue
            d = form.lower() if lowercase else form
            h = sent[head - 1][0]
            h = h.lower() if lowercase else h
            counts['dep'][rel][d] += 1
            counts['head'][rel][h] += 1
    return counts


def type_vectors_contextual(conllx, emb_path, align_path, layer, lowercase):
    """Mean contextual vector per word type, using the probe's own alignment."""
    import h5py
    import torch

    sums, freqs, dim = {}, defaultdict(int), None
    with h5py.File(emb_path, 'r') as ef, h5py.File(align_path, 'r') as af:
        n_pre = int(af.attrs.get('n_prefix_special', 1))
        n_suf = int(af.attrs.get('n_suffix_special', 1))
        for idx, sent in enumerate(parse_conllx(conllx)):
            key = str(idx)
            if key not in ef:
                break
            feats = ef[key][layer]                       # (n_subwords, hidden)
            align = np.asarray(af[key])                  # (n_sub_no_specials, n_words)
            end = feats.shape[0] - n_suf
            sub = torch.tensor(np.asarray(feats[n_pre:end]), dtype=torch.float)
            words = (torch.tensor(align, dtype=torch.float).t() @ sub).numpy()
            if words.shape[0] != len(sent):
                print(f'  warning: sentence {idx} aligned to {words.shape[0]} words '
                      f'but conllx has {len(sent)}; skipping', file=sys.stderr)
                continue
            if dim is None:
                dim = words.shape[1]
            for (form, _, _), vec in zip(sent, words):
                w = form.lower() if lowercase else form
                if w not in sums:
                    sums[w] = np.zeros(dim, dtype=np.float64)
                sums[w] += vec
                freqs[w] += 1
            if (idx + 1) % 5000 == 0:
                print(f'  {idx + 1} sentences', flush=True)

    vocab = sorted(sums)
    mat = np.stack([sums[w] / freqs[w] for w in vocab]).astype(np.float32)
    return {w: i for i, w in enumerate(vocab)}, mat


def type_vectors_static(vec_path, needed, lowercase):
    """Load static vectors (.vec text format) for the words we actually need."""
    want = {(w.lower() if lowercase else w) for w in needed}
    vocab, rows = {}, []
    with open(vec_path, encoding='utf-8', errors='ignore') as fh:
        first = fh.readline().split()
        dim = int(first[1]) if len(first) == 2 else None
        if dim is None:
            fh.seek(0)
        for line in fh:
            parts = line.rstrip().split(' ')
            w = parts[0]
            if w not in want or w in vocab:
                continue
            vocab[w] = len(rows)
            rows.append(np.asarray(parts[1:], dtype=np.float32))
    if not rows:
        sys.exit('no vectors matched the corpus vocabulary')
    print(f'  {len(vocab)}/{len(want)} types found ({len(want) - len(vocab)} OOV)')
    return vocab, np.stack(rows)


def pca_reduce(mat, n_components):
    """Exact PCA by eigendecomposition of the d x d covariance matrix.

    The earlier implementation used sklearn's randomized_svd, which is an
    approximation: its basis depends on the random projection and on row order,
    so entropies shift by ~0.03 bits between runs on the same data (verified
    against this implementation -- with PCA disabled the two agree to 1e-6, i.e.
    to the printed precision, so the entropy maths is identical and the whole
    discrepancy comes from the randomised basis).

    Since d is only ~300-4096, the covariance is small and a full symmetric
    eigendecomposition is both exact and cheap, which makes the reported
    numbers deterministic and reproducible. For X centred, X = U S V^T gives
    covariance V S^2 V^T, so projecting X onto the leading eigenvectors of the
    covariance reproduces U S exactly.
    """
    if n_components <= 0 or n_components >= mat.shape[1]:
        return mat
    centred = (mat - mat.mean(axis=0, keepdims=True)).astype(np.float64)
    cov = centred.T @ centred                       # (d, d), exact
    eigvals, eigvecs = np.linalg.eigh(cov)          # ascending, deterministic
    top = eigvecs[:, ::-1][:, :n_components]        # leading components
    # Sign convention: make each component's largest-magnitude entry positive,
    # so the basis is unique (eigenvectors are defined only up to sign).
    signs = np.sign(top[np.argmax(np.abs(top), axis=0), np.arange(n_components)])
    signs[signs == 0] = 1.0
    top = top * signs
    var = eigvals[::-1][:n_components].sum() / eigvals.sum()
    print(f'  PCA {mat.shape[1]}d -> {n_components}d (exact; '
          f'{var:.1%} of variance retained)')
    return (centred @ top).astype(np.float32)


def l2_normalise(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def shannon_entropy(counts):
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def sim_entropy(counts, word2idx, vectors, chunk):
    """-sum_x p(x) log2 q(x) with q = max(cos,0) @ p, computed in row chunks.

    Returns (H_sim, zpp) where zpp = sum_x sum_y Z_xy p(x) p(y) is the mean
    within-relation similarity. zpp is not used by the published entropy, which
    is the formula printed in the paper; it is reported so that the normalised
    variant -sum_x p(x) log2 (q(x)/zpp) = H_sim + log2(zpp) can be fitted and
    compared. See compare_diversity_normalisation.py: the two differ by a term
    that varies across relations, so which one predicts better is an empirical
    question rather than a matter of convention.

    Types with no vector are treated as orthogonal to everything (q(x) = p(x)),
    which is the same convention the sparse implementation uses for missing pairs.
    """
    words = list(counts)
    total = sum(counts.values())
    if total == 0 or not words:
        return 0.0, 1.0
    probs = np.array([counts[w] / total for w in words], dtype=np.float32)

    idx = np.array([word2idx.get(w, -1) for w in words])
    have = np.where(idx >= 0)[0]
    q = probs.copy()                                   # self term, cos(x,x)=1
    if len(have) > 1:
        V = vectors[idx[have]]                         # (m, d), already L2-normed
        p_have = probs[have]
        out = np.empty(len(have), dtype=np.float32)
        for a in range(0, len(have), chunk):
            b = min(a + chunk, len(have))
            S = V[a:b] @ V.T                           # (chunk, m) cosines
            np.maximum(S, 0.0, out=S)
            np.fill_diagonal(S[:, a:b], 0.0)           # self counted already
            out[a:b] = S @ p_have
        q[have] += out
    q = np.maximum(q, 1e-12)
    zpp = float(np.sum(probs * q))
    return float(-np.sum(probs * np.log2(q))), zpp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('corpus', help='PTB CoNLL-X / CoNLL-U file')
    ap.add_argument('--out', required=True)
    ap.add_argument('--embeddings', help='per-checkpoint embeddings hdf5')
    ap.add_argument('--alignments', help='alignment matrices hdf5')
    ap.add_argument('--layer', type=int, help='checkpoint index within --embeddings')
    ap.add_argument('--vec', help='static .vec file (alternative to --embeddings)')
    ap.add_argument('--pca-dim', type=int, default=0)
    ap.add_argument('--lowercase', action='store_true')
    ap.add_argument('--chunk', type=int, default=2048)
    args = ap.parse_args()

    if bool(args.embeddings) == bool(args.vec):
        sys.exit('give exactly one of --embeddings (with --alignments/--layer) or --vec')

    print('Reading corpus...')
    sentences = list(parse_conllx(args.corpus))
    counts = relation_vocabularies(sentences, args.lowercase)
    relations = sorted(set(counts['head']) | set(counts['dep']))
    print(f'  {len(sentences)} sentences, {len(relations)} relations')

    if args.embeddings:
        if not (args.alignments and args.layer is not None):
            sys.exit('--embeddings requires --alignments and --layer')
        print(f'Building type vectors from checkpoint {args.layer}...')
        word2idx, vectors = type_vectors_contextual(
            args.corpus, args.embeddings, args.alignments, args.layer, args.lowercase)
        source = f'contextual:{args.embeddings}:layer{args.layer}'
    else:
        needed = set()
        for role in ('head', 'dep'):
            for rel in counts[role]:
                needed |= set(counts[role][rel])
        print(f'Loading static vectors for {len(needed)} types...')
        word2idx, vectors = type_vectors_static(args.vec, needed, args.lowercase)
        source = f'static:{args.vec}'

    if args.pca_dim:
        vectors = pca_reduce(vectors, args.pca_dim)
    vectors = l2_normalise(vectors)
    print(f'  vectors: {vectors.shape}')

    print('Computing per-relation entropies...')
    with open(args.out, 'w') as fout:
        fout.write('deprel\tn_distinct_heads\tn_distinct_deps\thead_entropy_bits\t'
                   'dep_entropy_bits\thead_sim_entropy_bits\tdep_sim_entropy_bits\t'
                   'head_zpp\tdep_zpp\n')
        for rel in relations:
            h, d = counts['head'][rel], counts['dep'][rel]
            h_ent, h_zpp = sim_entropy(h, word2idx, vectors, args.chunk)
            d_ent, d_zpp = sim_entropy(d, word2idx, vectors, args.chunk)
            fout.write('\t'.join([
                rel, str(len(h)), str(len(d)),
                f'{shannon_entropy(h):.6f}', f'{shannon_entropy(d):.6f}',
                f'{h_ent:.6f}', f'{d_ent:.6f}',
                f'{h_zpp:.8f}', f'{d_zpp:.8f}',
            ]) + '\n')
    print(f'Wrote {args.out}  (similarity source: {source})')


if __name__ == '__main__':
    main()
