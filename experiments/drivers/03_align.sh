#!/bin/bash
# Precompute subword-to-PTB alignment matrices for one model's tokenizer.
#
#   RUN=gptj sbatch experiments/drivers/03_align.sh
#
# Independent of 01/02 -- alignments depend only on the tokenizer, so models
# sharing one (GPT-2 and GPT-J do not; RoBERTa-Shuffle-n1 uses roberta-base's)
# can share an alignment directory. CPU only.
#SBATCH --partition=short
#SBATCH --job-name=align
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm-logs/align-%x-%j.log
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
source "$REPO/experiments/drivers/common.sh"

mkdir -p "$ALIGN_DIR"
for SPLIT in train dev test; do
  OUT="$ALIGN_DIR/alignments.$SPLIT.hdf5"
  [ -s "$OUT" ] && { echo "$SPLIT already aligned, skipping"; continue; }
  $PYTHON "$REPO/scripts/precompute_alignments_hf.py" \
    "$CORPUS/ptb3-wsj-$SPLIT.conllx" "$OUT" --model-name "$TOKENIZER"
done

# The special-token counts the script just recorded must be what the manifest
# declares; a mismatch means the probe would strip the wrong rows.
echo "=== checking alignments against the manifest ==="
$PYTHON "$REPO/scripts/check_embeddings.py" --run "$RUN" --tokenizer-check
