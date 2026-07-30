#!/usr/bin/env python3
"""
Sparse FastText Cosine Similarity for UD Relation Vocabularies
===============================================================
For each dependency relation, computes pairwise cosine similarity only among
words that appear in the same role (head or dependent) for that relation.
This is vastly more efficient than a full V×V matrix.

The result is saved as a scipy sparse matrix (CSR format) covering only the
pairs that are actually needed for similarity-weighted entropy computation.

Algorithm
---------
1. Parse the CoNLL-U corpus; record, for each relation:
     heads(r)  = set of word forms that appear as head of r
     deps(r)   = set of word forms that appear as dependent of r
2. Build a global vocabulary from the union of all heads and deps.
3. Load fastText embeddings for every word in the vocabulary.
4. For each relation, compute pairwise cosine similarity within
   heads(r) and within deps(r).
5. Accumulate all (i, j, sim) triples and build a sparse CSR matrix.

The number of pairs is Σ_r [C(|heads_r|,2) + C(|deps_r|,2)], which for
EWT is typically ~1–5 M pairs versus ~145 M for the full 17k×17k matrix.

Usage
-----
  python fasttext_similarity_sparse.py corpus.conllu \\
      --model cc.en.300.bin \\
      --output ewt_sparse_sim

Output
------
  <output>.npz       -- scipy sparse CSR matrix (word_i × word_j → cosine sim)
  <output>.vocab.txt -- one word per line, index matches matrix rows/cols
  <output>.relations.npz -- per-relation head/dep word index sets

Dependencies
------------
  pip install fasttext-wheel numpy scipy tqdm
"""

import argparse
import sys
import os
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


# ---------------------------------------------------------------------------
# PCA dimensionality reduction
# ---------------------------------------------------------------------------

def pca_reduce(matrix: np.ndarray, n_components: int, topn: int = 0) -> np.ndarray:
    """
    Reduce embedding matrix via PCA, optionally removing the first `topn`
    principal components and keeping the next `n_components` components.
    Uses SVD on the mean-centred matrix; no external library required
    (falls back from sklearn's randomized SVD to numpy's full SVD).
    Returns a new float32 matrix of shape (V, n_components).
    """
    if n_components <= 0:
        raise ValueError("--pca-dim must be a positive integer.")
    if topn < 0:
        raise ValueError("-topn/--topn must be >= 0.")

    emb_dim = matrix.shape[1]
    if topn + n_components > emb_dim:
        raise ValueError(
            f"Requested components exceed embedding dim: topn({topn}) + "
            f"pca-dim({n_components}) > {emb_dim}."
        )

    if topn == 0 and n_components >= emb_dim:
        print(f"  --pca-dim {n_components} >= embedding dim {emb_dim}; skipping PCA.",
              file=sys.stderr)
        return matrix

    total_components = topn + n_components
    print(
        f"Reducing {emb_dim}d via PCA: removing top {topn} PCs, "
        f"keeping next {n_components} (components {topn}..{total_components - 1}).",
        file=sys.stderr,
    )
    mean    = matrix.mean(axis=0)
    centred = (matrix - mean).astype(np.float64)   # SVD more stable in float64

    try:
        from sklearn.utils.extmath import randomized_svd
        if total_components < min(centred.shape):
            U, s, Vt = randomized_svd(centred, n_components=total_components, random_state=42)
        else:
            raise ImportError
    except ImportError:
        # numpy full SVD — slower for large (V, 300) matrices but no extra deps
        U, s, Vt = np.linalg.svd(centred, full_matrices=False)
        U  = U[:, :total_components]
        s  = s[:total_components]

    U_keep = U[:, topn:topn + n_components]
    s_keep = s[topn:topn + n_components]

    reduced = (U_keep * s_keep).astype(np.float32)   # shape (V, n_components)

    # Report variance explained
    total_var = float((centred ** 2).sum())
    removed_var = float((s[:topn] ** 2).sum()) if topn > 0 else 0.0
    kept_var  = float((s_keep ** 2).sum())
    if topn > 0:
        print(f"  Variance removed by top {topn} PCs: "
              f"{100 * removed_var / total_var:.1f}%", file=sys.stderr)
    print(f"  Variance explained by kept PCs ({topn}..{topn + n_components - 1}): "
          f"{100 * kept_var / total_var:.1f}%", file=sys.stderr)

    return reduced


# ---------------------------------------------------------------------------
# CoNLL-U parsing
# ---------------------------------------------------------------------------

def parse_relation_vocabularies(conllu_path: str, lowercase: bool = False
                                ) -> tuple[dict, dict]:
    """
    Returns:
        head_sets : {deprel: set of word forms appearing as head}
        dep_sets  : {deprel: set of word forms appearing as dependent}
    """
    head_sets: dict[str, set] = defaultdict(set)
    dep_sets:  dict[str, set] = defaultdict(set)

    sentence = []
    with open(conllu_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if sentence:
                    _process_sentence(sentence, head_sets, dep_sets, lowercase)
                    sentence = []
                continue
            fields = line.split("\t")
            if len(fields) < 8:
                continue
            tid = fields[0]
            if "-" in tid or "." in tid:
                continue
            sentence.append({
                "id":     int(tid),
                "form":   fields[1].lower() if lowercase else fields[1],
                "head":   fields[6],
                "deprel": fields[7].lower(),
            })
    if sentence:
        _process_sentence(sentence, head_sets, dep_sets, lowercase)

    return dict(head_sets), dict(dep_sets)


def _process_sentence(sentence, head_sets, dep_sets, lowercase):
    id2form = {0: "<ROOT>"}
    for tok in sentence:
        id2form[tok["id"]] = tok["form"]

    for tok in sentence:
        deprel = tok["deprel"]
        dep_form = tok["form"]
        head_str = tok["head"]

        if not head_str.lstrip("-").isdigit():
            continue
        head_id = int(head_str)
        if head_id not in id2form:
            continue

        head_form = id2form[head_id]
        dep_sets[deprel].add(dep_form)
        head_sets[deprel].add(head_form)


# ---------------------------------------------------------------------------
# Embedding loading
# ---------------------------------------------------------------------------

def load_embeddings(vocab: list[str], model_path: str
                    ) -> tuple[np.ndarray, list[str]]:
    """
    Load fastText embeddings for all words in vocab.

    Supported formats (auto-detected by extension):
      .bin  -- fastText native binary, loaded via the `fasttext` library.
               Supports subword OOV: every word gets a vector.
               Install: pip install fasttext-wheel
      .vec  -- fastText text format (also works for word2vec .vec/.txt).
               Words not in the file are silently skipped (no OOV support).
               No extra library needed beyond numpy.
      .bin (gensim fallback) -- if the `fasttext` library fails, the script
               will try gensim's KeyedVectors loader as a fallback.
               Install: pip install gensim

    Returns:
        matrix : float32 ndarray (len(found), dim), L2-normalised
        found  : list of words in the same order as matrix rows
    """
    ext = os.path.splitext(model_path)[1].lower()

    if ext == ".bin":
        matrix, found = _load_bin(vocab, model_path)
    else:
        matrix, found = _load_vec(vocab, model_path)

    # L2-normalise in place
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix /= norms

    return matrix, found


def _load_bin(vocab: list[str], model_path: str) -> tuple[np.ndarray, list[str]]:
    """Load fastText .bin model. Falls back to gensim if fasttext library fails."""

    # --- Attempt 1: official fasttext library ---
    try:
        import fasttext as ft_lib
    except ImportError:
        print("INFO: 'fasttext' library not found; trying gensim fallback.", file=sys.stderr)
        print("      To use the native loader: pip install fasttext-wheel", file=sys.stderr)
        return _load_bin_gensim(vocab, model_path)

    print(f"Loading fastText binary model: {model_path}", file=sys.stderr)
    print("(this takes ~30 s and ~4 GB RAM for cc.en.300.bin)", file=sys.stderr)
    try:
        ft = ft_lib.load_model(model_path)
    except ValueError as e:
        print(f"\nfasttext failed to open the model: {e}", file=sys.stderr)
        print("Common causes:", file=sys.stderr)
        print("  1. Incomplete download — check: ls -lh", model_path, file=sys.stderr)
        print("     Expected size: ~7 GB for cc.en.300.bin", file=sys.stderr)
        print("  2. Wrong package — uninstall any 'fasttext' variants and run:", file=sys.stderr)
        print("       pip uninstall fasttext fasttext-wheel fasttext-langdetect", file=sys.stderr)
        print("       pip install fasttext-wheel", file=sys.stderr)
        print("  3. Use the .vec text format instead (no binary parser needed):", file=sys.stderr)
        print("     Download cc.en.300.vec.gz, gunzip it, pass the .vec file.", file=sys.stderr)
        print("\nAttempting gensim fallback...", file=sys.stderr)
        return _load_bin_gensim(vocab, model_path)

    dim = ft.get_dimension()
    matrix = np.zeros((len(vocab), dim), dtype=np.float32)
    for i, word in enumerate(tqdm(vocab, desc="Fetching vectors", unit="w")):
        matrix[i] = ft.get_word_vector(word)  # always returns a vector (subword)
    return matrix, list(vocab)


def _load_bin_gensim(vocab: list[str], model_path: str) -> tuple[np.ndarray, list[str]]:
    """Gensim fallback for .bin files (no subword OOV support)."""
    try:
        from gensim.models import FastText as GensimFT
        from gensim.models import KeyedVectors
    except ImportError:
        sys.exit(
            "Neither 'fasttext' nor 'gensim' is available.\n"
            "Install one of:\n"
            "  pip install fasttext-wheel\n"
            "  pip install gensim\n"
            "Or use the .vec text format instead."
        )

    print(f"Loading with gensim: {model_path}", file=sys.stderr)
    try:
        # Try native fastText format first
        model = GensimFT.load_fasttext_format(model_path)
        kv = model.wv
    except Exception:
        # Try word2vec binary format
        kv = KeyedVectors.load_word2vec_format(model_path, binary=True)

    vocab_set = set(vocab)
    found, vecs = [], []
    for w in vocab:
        if w in kv:
            found.append(w)
            vecs.append(kv[w])
        elif hasattr(kv, 'get_vector'):
            # gensim FastText supports OOV via subwords
            try:
                found.append(w)
                vecs.append(kv.get_vector(w))
            except KeyError:
                pass

    if not found:
        sys.exit("No vocabulary words found in the model. Check model format and path.")

    matrix = np.stack(vecs).astype(np.float32)
    missing = len(vocab) - len(found)
    if missing:
        print(f"  {missing} words not found in gensim model (no subword fallback).", file=sys.stderr)
    return matrix, found


def _load_vec(vocab: list[str], model_path: str) -> tuple[np.ndarray, list[str]]:
    """Load fastText/word2vec .vec text format. Scans file once, picks up vocab words."""
    print(f"Loading text vectors: {model_path}", file=sys.stderr)
    vocab_set = set(vocab)
    vectors: dict[str, np.ndarray] = {}

    with open(model_path, encoding="utf-8", errors="ignore") as fh:
        header = fh.readline().strip().split()
        # Some .vec files have a header (n_words dim), some don't
        if len(header) == 2 and header[0].isdigit() and header[1].isdigit():
            pass  # normal header, already consumed
        else:
            # No header — first line is a word vector; reprocess it
            parts = header
            if parts[0] in vocab_set:
                vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)

        for line in tqdm(fh, desc="Scanning .vec", unit=" lines"):
            parts = line.rstrip().split(" ")
            if not parts or parts[0] not in vocab_set:
                continue
            try:
                vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            if len(vectors) == len(vocab_set):
                break  # found everything we need

    found   = [w for w in vocab if w in vectors]
    missing = [w for w in vocab if w not in vectors]
    if missing:
        print(f"  {len(missing)} words not found in .vec file (no subword support).", file=sys.stderr)
        print(f"  Use a .bin model for full OOV coverage.", file=sys.stderr)
    if not found:
        sys.exit("No vocabulary words found in the model file. Check the path and format.")

    matrix = np.stack([vectors[w] for w in found]).astype(np.float32)
    return matrix, found


# ---------------------------------------------------------------------------
# Sparse similarity computation
# ---------------------------------------------------------------------------

def compute_sparse_similarity(
        head_sets: dict,
        dep_sets:  dict,
        word2idx:  dict[str, int],
        normed:    np.ndarray,
) -> tuple[sp.csr_matrix, dict]:
    """
    For each relation, compute pairwise cosine similarity within
    heads(r) and within deps(r). Accumulate into a sparse symmetric matrix.

    Returns:
        sim_sparse   : scipy.sparse.csr_matrix, shape (V, V)
        rel_idx_sets : {deprel: {"heads": [idx,...], "deps": [idx,...]}}
    """
    rows, cols, vals = [], [], []
    rel_idx_sets = {}
    all_relations = sorted(set(head_sets) | set(dep_sets))

    for deprel in tqdm(all_relations, desc="Relations", unit="rel"):
        h_words = sorted(head_sets.get(deprel, set()) & word2idx.keys())
        d_words = sorted(dep_sets.get(deprel, set())  & word2idx.keys())

        h_idx = [word2idx[w] for w in h_words]
        d_idx = [word2idx[w] for w in d_words]
        rel_idx_sets[deprel] = {"heads": h_idx, "deps": d_idx}

        for role_idx in (h_idx, d_idx):
            if len(role_idx) < 2:
                continue
            vecs = normed[role_idx]           # (m, dim)
            sim_block = vecs @ vecs.T         # (m, m), all pairs
            # Extract upper triangle (excluding diagonal)
            m = len(role_idx)
            for ii in range(m):
                for jj in range(ii + 1, m):
                    i_global = role_idx[ii]
                    j_global = role_idx[jj]
                    s = float(sim_block[ii, jj])
                    # Store both (i,j) and (j,i) for symmetric access
                    rows.append(i_global); cols.append(j_global); vals.append(s)
                    rows.append(j_global); cols.append(i_global); vals.append(s)

    V = normed.shape[0]
    # Duplicate entries are summed by default; use maximum instead via LIL
    # Build COO first, then convert. For duplicate (i,j), keep max similarity.
    # (Duplicates arise when the same word pair co-occurs across multiple relations)
    print("Building sparse matrix...", file=sys.stderr)
    coo = sp.coo_matrix(
        (np.array(vals, dtype=np.float32),
         (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(V, V)
    )
    # For duplicates, keep the maximum value
    csr = coo.tocsr()
    csr = _csr_keep_max_duplicates(coo, V)

    return csr, rel_idx_sets


def _csr_keep_max_duplicates(coo: sp.coo_matrix, V: int) -> sp.csr_matrix:
    """
    Convert COO to CSR, keeping the maximum value for duplicate (i,j) entries.

    Vectorized approach:
      1. Lexicographically sort all (row, col) pairs.
      2. Find the boundaries between runs of identical (row, col) pairs.
      3. Within each run take the max using np.maximum.reduceat.
      4. Build a new COO from the deduplicated triples.
    """
    rows, cols, vals = coo.row, coo.col, coo.data

    # 1. Sort by (row, col) lexicographically
    order = np.lexsort((cols, rows))
    rows_s = rows[order]
    cols_s = cols[order]
    vals_s = vals[order]

    # 2. Find the start of each new (row, col) group
    #    A new group starts where either row or col changes from the previous entry.
    if len(rows_s) == 0:
        return sp.csr_matrix((V, V), dtype=np.float32)

    change = np.ones(len(rows_s), dtype=bool)
    change[1:] = (rows_s[1:] != rows_s[:-1]) | (cols_s[1:] != cols_s[:-1])
    group_starts = np.where(change)[0]

    # 3. Take max within each group using reduceat
    max_vals = np.maximum.reduceat(vals_s, group_starts)

    # 4. The representative (row, col) for each group is the first entry
    dedup_rows = rows_s[group_starts]
    dedup_cols = cols_s[group_starts]

    # 5. Build the final CSR matrix from deduplicated triples
    csr = sp.csr_matrix(
        (max_vals.astype(np.float32), (dedup_rows, dedup_cols)),
        shape=(V, V)
    )
    return csr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sparse fastText cosine similarity for UD relation vocabularies."
    )
    parser.add_argument("corpus",  help="CoNLL-U corpus file")
    parser.add_argument("--model", required=True,
                        help="Path to fastText model (.bin or .vec)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file prefix (default: derived from corpus name)")
    parser.add_argument("--lowercase", action="store_true",
                        help="Lowercase all word forms")
    parser.add_argument("--pca-dim", type=int, default=None,
                        help="Reduce embeddings to this many dimensions via PCA "
                             "before computing similarity (e.g. 50 or 100). "
                             "Recommended: 50-100 for similarity-weighted entropy. "
                             "Default: use full embedding dimensionality.")
    parser.add_argument("-topn", "--topn", type=int, default=0,
                        help="Remove the top N principal components, then keep "
                             "the next --pca-dim components. Requires --pca-dim.")
    args = parser.parse_args()

    if args.topn < 0:
        parser.error("-topn/--topn must be >= 0.")
    if args.topn > 0 and args.pca_dim is None:
        parser.error("-topn/--topn requires --pca-dim.")

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.corpus))[0]
        args.output = base

    # Step 1: collect per-relation vocabularies
    print(f"Parsing corpus: {args.corpus}", file=sys.stderr)
    head_sets, dep_sets = parse_relation_vocabularies(args.corpus, args.lowercase)

    all_words = sorted(
        set().union(*head_sets.values(), *dep_sets.values())
    )
    print(f"Total unique words across all relation roles: {len(all_words):,}", file=sys.stderr)

    n_pairs = sum(
        len(h) * (len(h) - 1) // 2 + len(d) * (len(d) - 1) // 2
        for h, d in zip(head_sets.values(), dep_sets.values())
    )
    print(f"Total pairs to compute: {n_pairs:,}  "
          f"(full matrix would be {len(all_words)**2:,})", file=sys.stderr)

    # Step 2: load embeddings
    matrix, found = load_embeddings(all_words, args.model)
    print(f"Vocabulary with embeddings: {len(found):,}  "
          f"(embedding dim: {matrix.shape[1]})", file=sys.stderr)

    # Step 2b: optional PCA reduction
    if args.pca_dim is not None:
        matrix = pca_reduce(matrix, args.pca_dim, topn=args.topn)
        # Re-normalise after PCA (PCA does not preserve L2 norm)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix /= norms
        print(f"  Re-normalised after PCA. Final dim: {matrix.shape[1]}", file=sys.stderr)

    word2idx = {w: i for i, w in enumerate(found)}

    # Step 3: compute sparse similarity
    sim_sparse, rel_idx_sets = compute_sparse_similarity(
        head_sets, dep_sets, word2idx, matrix
    )
    nnz = sim_sparse.nnz
    print(f"Non-zero entries in sparse matrix: {nnz:,}  "
          f"({100*nnz/len(found)**2:.3f}% of full matrix)", file=sys.stderr)

    # Step 4: save
    vocab_path = f"{args.output}.vocab.txt"
    sim_path   = f"{args.output}.sim_sparse.npz"
    rel_path   = f"{args.output}.relations.npz"

    with open(vocab_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(found) + "\n")
    print(f"Vocab saved: {vocab_path}", file=sys.stderr)

    sp.save_npz(sim_path, sim_sparse)
    print(f"Sparse similarity matrix saved: {sim_path}", file=sys.stderr)

    # Save relation index sets as a numpy archive
    rel_save = {}
    for rel, d in rel_idx_sets.items():
        safe = rel.replace(":", "_")
        rel_save[f"{safe}__heads"] = np.array(d["heads"], dtype=np.int32)
        rel_save[f"{safe}__deps"]  = np.array(d["deps"],  dtype=np.int32)
    np.savez_compressed(rel_path, **rel_save)
    print(f"Relation index sets saved: {rel_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Loading utilities for downstream use
# ---------------------------------------------------------------------------



def load_sparse_sim(output_prefix: str):
    """
    Load saved sparse similarity data.
    Returns:
        vocab    : list of word strings
        sim      : scipy.sparse.csr_matrix
        rel_sets : {deprel: {"heads": [idx,...], "deps": [idx,...]}}
    """
    with open(f"{output_prefix}.vocab.txt", encoding="utf-8") as fh:
        vocab = [line.rstrip("\n") for line in fh]

    sim = sp.load_npz(f"{output_prefix}.sim_sparse.npz")

    raw = np.load(f"{output_prefix}.relations.npz", allow_pickle=True)
    rel_sets = {}
    for key in raw.files:
        rel, role = key.rsplit("__", 1)
        rel = rel.replace("_", ":", 1) if "_" in rel else rel
        if rel not in rel_sets:
            rel_sets[rel] = {}
        rel_sets[rel][role] = list(raw[key])

    return vocab, sim, rel_sets


if __name__ == "__main__":
    main()
