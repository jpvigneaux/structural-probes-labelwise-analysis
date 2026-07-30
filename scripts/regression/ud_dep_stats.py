#!/usr/bin/env python3
"""
UD CoNLL Dependency Relation Statistics
========================================
Reads a CoNLL-U formatted Universal Dependencies corpus and computes,
for each dependency relation:
  - number of distinct words appearing as head
  - number of distinct words appearing as dependent
  - entropy of the head distribution (standard Shannon entropy)
  - entropy of the dependent distribution (standard Shannon entropy)
  - similarity-weighted entropy of the head distribution   [optional]
  - similarity-weighted entropy of the dependent distribution [optional]

Similarity-weighted entropy (also called neighbourhood entropy) is:

    H_sim(r, role) = -Σ_w  p(w) · log q(w)

where:
    p(w)  = empirical probability of word w in role 'role' for relation r
    q(w)  = Σ_{w'} p(w') · sim(w, w') / Z(r, role)
    Z     = Σ_w Σ_{w'} p(w') · sim(w, w')   (normalisation constant)
    sim   = cosine similarity from the sparse matrix produced by
            fasttext_similarity_sparse.py

Words that are semantically similar share probability mass under q, so a
distribution concentrated on near-synonyms has lower H_sim than Shannon H.

The sparse similarity matrix only stores pairs that co-occur within the same
relation role, so words with no stored similarity to any other word in their
role are treated as orthogonal to all others (sim = 0 for missing pairs,
sim = 1 for self).

Usage
-----
    # Standard entropy only
    python ud_dep_stats.py corpus.conllu

    # With similarity-weighted entropy
    python ud_dep_stats.py corpus.conllu \\
        --sim-prefix ewt_50d \\
        [--lowercase] [--output out.tsv]

The --sim-prefix argument should be the output prefix passed to
fasttext_similarity_sparse.py (e.g. 'ewt_50d'), which produced:
    ewt_50d.vocab.txt
    ewt_50d.sim_sparse.npz
    ewt_50d.relations.npz

CoNLL-U format (tab-separated, 10 fields per token line):
    ID  FORM  LEMMA  UPOS  XPOS  FEATS  HEAD  DEPREL  DEPS  MISC
Blank lines separate sentences; lines starting with '#' are comments.
Multi-word tokens (ID like '1-2') and empty nodes (ID like '1.1') are skipped.
"""

import argparse
import math
import sys
from collections import defaultdict

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# CoNLL-U parsing
# ---------------------------------------------------------------------------

def parse_conllu(path: str):
    """Yield sentences as lists of token dicts (id, form, head, deprel)."""
    sentence = []
    with open(path, encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line == "":
                if sentence:
                    yield sentence
                    sentence = []
                continue
            fields = line.split("\t")
            if len(fields) < 8:
                continue
            token_id = fields[0]
            if "-" in token_id or "." in token_id:
                continue
            sentence.append({
                "id":     int(token_id),
                "form":   fields[1],
                "head":   fields[6],
                "deprel": fields[7].lower(),
            })
    if sentence:
        yield sentence


# ---------------------------------------------------------------------------
# Count collection
# ---------------------------------------------------------------------------

def collect_counts(corpus_path: str, lowercase: bool = False):
    """
    Return:
        vocab       : set of all word forms seen
        head_counts : {deprel: {word: count}}
        dep_counts  : {deprel: {word: count}}
    """
    vocab: set = set()
    head_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dep_counts:  dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sentence in parse_conllu(corpus_path):
        id2form: dict[int, str] = {0: "<ROOT>"}
        for tok in sentence:
            form = tok["form"].lower() if lowercase else tok["form"]
            id2form[tok["id"]] = form
            vocab.add(form)

        for tok in sentence:
            deprel   = tok["deprel"]
            dep_form = id2form[tok["id"]]
            head_str = tok["head"]
            if not head_str.lstrip("-").isdigit():
                continue
            head_id = int(head_str)
            if head_id in id2form:
                head_form = id2form[head_id]
                head_counts[deprel][head_form] += 1
                vocab.add(head_form)
            dep_counts[deprel][dep_form] += 1

    return vocab, head_counts, dep_counts


# ---------------------------------------------------------------------------
# Standard Shannon entropy
# ---------------------------------------------------------------------------

def shannon_entropy(counts: dict[str, int]) -> float:
    """H = -Σ p(w) log2 p(w) over words with count > 0."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Sparse similarity data loader
# ---------------------------------------------------------------------------

import sys
import numpy as np
import scipy.sparse as sp

# def load_sparse_sim(prefix: str, k: int = 10):
#     """
#     Load the sparse similarity matrix and apply Cross-Domain Similarity 
#     Local Scaling (CSLS) to correct for embedding hubness/heteroscedasticity.
    
#     Parameters:
#         prefix (str): Prefix path for vocab and matrix files.
#         k (int): Number of nearest neighbors to look at for local density scaling.
#     """
#     vocab_path = f"{prefix}.vocab.txt"
#     sim_path   = f"{prefix}.sim_sparse.npz"

#     # 1. Load Vocabulary Map
#     with open(vocab_path, encoding="utf-8") as fh:
#         vocab = [line.rstrip("\n") for line in fh if line.strip()]
#     word2idx = {w: i for i, w in enumerate(vocab)}

#     # 2. Load Raw Sparse Matrix
#     raw_sim_csr = sp.load_npz(sim_path)
#     print(f"Loaded raw sparse similarity matrix: {raw_sim_csr.shape}, "
#           f"{raw_sim_csr.nnz:,} non-zeros", file=sys.stderr)

#     # 3. Ensure format allows row manipulations and strip negative cosine values
#     sim_csr = raw_sim_csr.tocsr()
#     sim_csr.data = np.maximum(sim_csr.data, 0.0)

#     # 4. Compute r_K(w_i): Mean similarity to K-nearest neighbors for every word
#     print(f"Calculating local neighborhood density bias (K={k})...", file=sys.stderr)
#     num_words = sim_csr.shape[0]
#     r_K = np.zeros(num_words, dtype=np.float64)

#     for i in range(num_words):
#         row_start = sim_csr.indptr[i]
#         row_end   = sim_csr.indptr[i+1]
        
#         if row_end > row_start:
#             row_values = sim_csr.data[row_start:row_end]
#             # Select the top-K highest similarity links in the row
#             if len(row_values) > k:
#                 top_k_values = np.partition(row_values, -k)[-k:]
#                 r_K[i] = np.mean(top_k_values)
#             else:
#                 r_K[i] = np.mean(row_values)

#     # 5. Apply CSLS Transformation over Sparse Coordinates
#     print("Applying CSLS transformations across the sparse matrix...", file=sys.stderr)
    
#     # Unpack sparse vectors efficiently using coordinate mappings
#     sim_coo = sim_csr.tocoo()
#     rows, cols, data = sim_coo.row, sim_coo.col, sim_coo.data

#     # Apply CSLS formula
#     csls_data = (2.0 * data) - r_K[rows] - r_K[cols]

#     # Reconstruct the corrected sparse matrix
#     csls_csr = sp.coo_matrix((csls_data, (rows, cols)), shape=sim_csr.shape).tocsr()

#     # Enforce standard boundaries: Keep self-similarity at 1.0, clip negatives to 0.0
#     csls_csr.data = np.maximum(csls_csr.data, 0.0)
#     csls_csr.setdiag(1.0)
#     csls_csr.eliminate_zeros()

#     print(f"CSLS adjustment complete. Final active entries: {csls_csr.nnz:,}", file=sys.stderr)
#     return word2idx, csls_csr


def load_sparse_sim(prefix: str):
    """
    Load the three files produced by fasttext_similarity_sparse.py.
    Returns:
        word2idx : {word: int}
        sim_csr  : scipy.sparse.csr_matrix  (V x V, cosine similarities)
    """
    vocab_path = f"{prefix}.vocab.txt"
    sim_path   = f"{prefix}.sim_sparse.npz"

    with open(vocab_path, encoding="utf-8") as fh:
        vocab = [line.rstrip("\n") for line in fh if line.strip()]
    word2idx = {w: i for i, w in enumerate(vocab)}

    sim_csr = sp.load_npz(sim_path)
    print(f"Loaded sparse similarity matrix: {sim_csr.shape}, "
          f"{sim_csr.nnz:,} non-zeros", file=sys.stderr)
    return word2idx, sim_csr


# ---------------------------------------------------------------------------
# Similarity-weighted entropy
# ---------------------------------------------------------------------------

def sim_weighted_entropy_fast(counts: dict[str, int],
                               word2idx: dict[str, int],
                               sim_csr: sp.csr_matrix) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0

    words   = list(counts.keys())
    probs   = np.array([counts[w] / total for w in words], dtype=np.float64)
    indices = np.array([word2idx.get(w, -1) for w in words], dtype=np.int64)

    valid_pos = np.where(indices >= 0)[0]
    valid_idx = indices[valid_pos]

    # 1. Initialize q with self-similarity: q(w) = p(w) * 1
    q = probs.copy()

    if len(valid_idx) > 1:
        # Extract the (n_valid x n_valid) submatrix
        sub_sq = sim_csr[valid_idx, :][:, valid_idx].toarray()

        # Clip negative similarities to 0
        sub_sq = np.maximum(sub_sq, 0.0)

        # FIX BUG 1: Zero out the diagonal so we don't double-count self-similarity
        np.fill_diagonal(sub_sq, 0.0)

        # Extract only the probabilities of valid words
        valid_probs = probs[valid_pos]

        # Compute cross-similarity contributions: Σ_{w' ≠ w} p(w') * sim(w, w')
        # Matrix multiplication handles the sum across rows/columns correctly
        cross_sim_contrib = sub_sq @ valid_probs

        # Add cross-commonalities to our self-similarity baseline
        q[valid_pos] += cross_sim_contrib

    # FIX BUG 2: Do NOT divide by Z.
    # Clip q to a tiny epsilon above 0 to prevent log2(0) errors if any float issues occur
    q = np.maximum(q, 1e-12)

    # Compute LCR framework / Neighborhood Entropy: -Σ p(w) * log2(q(w))
    h_sim = -np.sum(probs * np.log2(q))
    
    return float(h_sim)


# def sim_weighted_entropy_fast(counts: dict[str, int],
#                                word2idx: dict[str, int],
#                                sim_csr: sp.csr_matrix) -> float:
#     """
#     Vectorised similarity-weighted (neighbourhood) entropy.

#     H_sim = -Σ_w p(w) · log2 q(w)

#     where q(w) = [Σ_{w'} p(w') · sim+(w, w')] / Z
#     and   sim+(w, w') = max(sim(w, w'), 0)  for w' ≠ w   (negative similarities
#                                                               treated as 0)
#           sim+(w, w)  = 1                                    (self-similarity)
#           Z           = Σ_w q_unnorm(w)                     (normalisation)

#     Clipping negative cosine similarities to 0 avoids q(w) going negative
#     (which would cause log(0) = -inf). Words with no positive-similarity
#     neighbours still receive q(w) >= p(w)/Z > 0 from their self-similarity.

#     Words absent from the similarity vocabulary are treated as having
#     sim = 0 with all other words (only self-similarity contributes).
#     """
#     total = sum(counts.values())
#     if total == 0:
#         return float("nan")

#     words   = list(counts.keys())
#     probs   = np.array([counts[w] / total for w in words], dtype=np.float64)
#     indices = np.array([word2idx.get(w, -1) for w in words], dtype=np.int64)

#     valid_pos = np.where(indices >= 0)[0]
#     valid_idx = indices[valid_pos]

#     # q initialised with self-similarity contribution: q[i] = p[i] * 1
#     q = probs.copy()

#     if len(valid_idx) > 1:
#         # Extract the (n_valid x n_valid) submatrix of stored similarities
#         sub_sq = sim_csr[valid_idx, :][:, valid_idx].toarray()  # (n_valid, n_valid)

#         # Clip negatives to 0 — negative cosine similarity means "dissimilar",
#         # not "opposite-probability"; treating it as 0 is semantically correct.
#         np.clip(sub_sq, 0.0, None, out=sub_sq)

#         # Zero diagonal (self-sim already in q via initialisation)
#         np.fill_diagonal(sub_sq, 0.0)

#         # q[valid_pos[i]] += Σ_j p[valid_pos[j]] * sim+(i, j)
#         probs_valid = probs[valid_pos]
#         q[valid_pos] += sub_sq @ probs_valid

#     # Normalise to a proper distribution
#     Z = q.sum()
#     if Z <= 0:
#         return float("nan")
#     q /= Z

#     # H_sim = -Σ p log2 q  (q[i] >= p[i]/Z > 0 for all i, so log is safe)
#     nonzero = probs > 0
#     h_sim = -float(np.sum(probs[nonzero] * np.log2(q[nonzero])))
#     return h_sim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute per-relation statistics from a CoNLL-U corpus, "
                    "including optional similarity-weighted entropy."
    )
    parser.add_argument("corpus", help="Path to the CoNLL-U file")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Write TSV output to this file (default: stdout)",
    )
    parser.add_argument(
        "--lowercase", "-l", action="store_true",
        help="Lowercase all word forms before counting",
    )
    parser.add_argument(
        "--sim-prefix", default=None,
        metavar="PREFIX",
        help="File prefix for sparse similarity data produced by "
             "fasttext_similarity_sparse.py (e.g. 'ewt_50d'). "
             "Enables similarity-weighted entropy columns. "
             "If omitted, only standard Shannon entropy is computed.",
    )
    args = parser.parse_args()

    # --- Load corpus counts ---
    print(f"Reading corpus: {args.corpus}", file=sys.stderr)
    vocab, head_counts, dep_counts = collect_counts(args.corpus, args.lowercase)
    vocab_size = len(vocab)
    print(f"Vocabulary size : {vocab_size:,}", file=sys.stderr)
    print(f"Relations found : {len(head_counts):,}", file=sys.stderr)

    # --- Optionally load similarity data ---
    word2idx = None
    sim_csr  = None
    if args.sim_prefix:
        print(f"Loading similarity data: {args.sim_prefix}.*", file=sys.stderr)
        word2idx, sim_csr = load_sparse_sim(args.sim_prefix)

    # --- Compute statistics per relation ---
    all_relations = sorted(set(head_counts) | set(dep_counts))
    use_sim = sim_csr is not None

    header_cols = [
        "deprel",
        "relation_count",
        "n_distinct_heads",
        "n_distinct_deps",
        "head_entropy_bits",
        "dep_entropy_bits",
    ]
    if use_sim:
        header_cols += ["head_sim_entropy_bits", "dep_sim_entropy_bits"]
    header = "\t".join(header_cols)

    rows = []
    for rel in all_relations:
        hc = head_counts.get(rel, {})
        dc = dep_counts.get(rel, {})

        rel_count = sum(dc.values())
        n_heads = len(hc)
        n_deps  = len(dc)
        h_head  = shannon_entropy(hc)
        h_dep   = shannon_entropy(dc)

        if use_sim:
            hs_head = sim_weighted_entropy_fast(hc, word2idx, sim_csr)
            hs_dep  = sim_weighted_entropy_fast(dc, word2idx, sim_csr)
            rows.append((rel, rel_count, n_heads, n_deps, h_head, h_dep, hs_head, hs_dep))
        else:
            rows.append((rel, rel_count, n_heads, n_deps, h_head, h_dep))

    rows.sort(key=lambda r: r[0])

    if use_sim:
        lines = [header] + [
            f"{rel}\t{rc}\t{nh}\t{nd}\t{hh:.6f}\t{hd:.6f}\t{hs_h:.6f}\t{hs_d:.6f}"
            for rel, rc, nh, nd, hh, hd, hs_h, hs_d in rows
        ]
    else:
        lines = [header] + [
            f"{rel}\t{rc}\t{nh}\t{nd}\t{hh:.6f}\t{hd:.6f}"
            for rel, rc, nh, nd, hh, hd in rows
        ]

    output_text = "\n".join(lines) + "\n"

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output_text)
        print(f"Results written to: {args.output}", file=sys.stderr)
    else:
        print(output_text, end="")


if __name__ == "__main__":
    main()
