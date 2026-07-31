#!/bin/bash
# Build the two corpus-statistics predictors. Run ONCE, not per model.
#
#   sbatch experiments/drivers/06_predictors.sh
#
# Both are computed on PTB *train*. Entropy estimated on the 1.7k-sentence dev
# split would be badly biased, and using train keeps the predictors independent
# of the split ULAS is measured on. They are identical for every row of the
# cross-model table -- only the ULAS values differ between models.
#
# 72G is measured, not guessed: the entropy step holds a dense per-relation
# block of 300-dimensional vectors for the largest relation. Memory requests
# count against the fairshare score, so do not round this up "to be safe".
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --job-name=predictors
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --output=slurm-logs/predictors-%j.log
set -euo pipefail
# Locate the repository. sbatch copies this script into the SLURM spool
# directory before running it, so its own path says nothing about where the repo
# is: prefer REPO_ROOT (exported by submit_all.sh), then the directory sbatch was
# invoked from, and fall back to this file's location for a direct shell run.
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[ -n "$REPO" ] && [ -f "$REPO/paths.sh" ] || \
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" || true
[ -f "$REPO/paths.sh" ] || {
  echo "cannot locate the repository (no paths.sh found). Submit from the repo" >&2
  echo "root, or export REPO_ROOT=/path/to/structural-probes-labelwise-analysis." >&2
  exit 1; }
source "$REPO/paths.sh"
export OMP_NUM_THREADS=4
DRV="$REPO/experiments/drivers"
M () { $PYTHON "$DRV/manifest.py" "$@"; }

CORPUS=$(M --root corpus)
OUT=$(M --root out)
PTB="$CORPUS/ptb3-wsj-train.conllx"
VEC=$(M --predictor fasttext_vec)
mkdir -p "$OUT"

echo "=== dependency-length moments per relation ==="
# Writes mean_length, mean_log_length, stdev_log_length and skew_log_length:
# the location, dispersion and asymmetry terms of the moment ladder.
$PYTHON "$REPO/scripts/regression/arc_length_moments.py" "$PTB" \
  --output "$OUT/dep-lengths-ptb.tsv"

echo "=== similarity-corrected head entropy, static fastText space ==="
# The earlier route built one global sparse matrix with an entry per
# within-relation word pair -- 756M of them on PTB train, which OOMs -- and its
# randomized-SVD PCA shifted entropies by ~0.03 bits between runs. This computes
# cosines in row chunks of a dense per-relation block, with an exact
# eigendecomposition, so it is both tractable and deterministic.
$PYTHON "$REPO/scripts/regression/contextual_sim_entropy.py" "$PTB" \
  --out "$OUT/sim_static_fasttext.tsv" --vec "$VEC" --pca-dim 50

echo "=== done ==="
head -3 "$OUT/dep-lengths-ptb.tsv" "$OUT/sim_static_fasttext.tsv"
