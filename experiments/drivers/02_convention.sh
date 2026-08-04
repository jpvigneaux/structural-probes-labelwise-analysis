#!/bin/bash
# Convert extracted checkpoints to the LayerNorm convention the run declares.
#
#   RUN=bertbase sbatch experiments/drivers/02_convention.sh
#
# Only runs declaring convention B need this. What "B" means differs by family
# and both cases are handled by apply_consuming_layernorm.py:
#
#   post-LN (BERT)   convert_raw_to_hf_all_checkpoints.py stores the PRE-LayerNorm
#                    accumulator, which is neither literature convention. Applying
#                    the consuming LayerNorm recovers the post-LayerNorm residual
#                    stream, i.e. Hewitt & Manning's `encoded_layers`.
#   pre-LN (GPT-2,   the residual stream is never normalised in place, so A (what
#   ModernBERT,      propagates) and B (what the next sub-layer reads) genuinely
#   GPT-J)           differ. The paper's runs for these use A.
#
# Runs declaring convention A, or extracted as post-LN hidden_states, exit here
# with nothing to do.
#SBATCH --partition=short
#SBATCH --job-name=convention
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-logs/convention-%x-%j.log
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

if [[ "$CONVENTION" != "B" ]]; then
  echo "run $RUN declares convention '$CONVENTION'; nothing to convert."
  exit 0
fi

# Convention B files live beside the A files under a -convB name; the manifest's
# `embeddings:` for such a run already points at the -convB directory.
SRC_DIR="${EMB_DIR%-convB}"
[ "$SRC_DIR" = "$EMB_DIR" ] && { echo "expected the manifest's embeddings dir to end in -convB for a convention-B run" >&2; exit 1; }
mkdir -p "$EMB_DIR"

for SPLIT in train dev test; do
  IN="$SRC_DIR/raw.$SPLIT.$STEM.hdf5"
  OUT="$EMB_DIR/raw.$SPLIT.$STEM.hdf5"
  [ -s "$OUT" ] && { echo "$SPLIT already converted, skipping"; continue; }
  [ -s "$IN" ] || { echo "missing input $IN -- run 01_extract.sh first" >&2; exit 1; }

  # --validate-sentences pre-hooks the consumer sub-layers and compares their
  # true inputs against the transform, so a wrong LayerNorm map fails here
  # rather than as a slightly worse probe fifty GPU-hours later.
  $PYTHON "$REPO/scripts/apply_consuming_layernorm.py" "$IN" "$OUT" "$HF_NAME" \
    --conllx "$CORPUS/ptb3-wsj-dev.conllx" --validate-sentences 25
done
echo "=== convention B written to $EMB_DIR ==="
