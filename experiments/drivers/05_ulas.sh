#!/bin/bash
# Per-relation UASL (dev and test) for every checkpoint of one run.
#
#   RUN=gpt2 RESULTS_ROOT=... sbatch --array=0-25 experiments/drivers/05_ulas.sh
#
# CPU only: this reads each probe's stored predictions, never the training
# embeddings. Writes dev.uuas_by_relation and test.uuas_by_relation into each
# layer-NN/, which is what every regression in the paper consumes.
#SBATCH --partition=short
#SBATCH --job-name=ulas
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=slurm-logs/ulas-%x-%A_%a.log
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

: "${RESULTS_ROOT:?set RESULTS_ROOT to the directory holding layer-NN/}"
LAYERS=${SLURM_ARRAY_TASK_ID:-0-$((N_CHECKPOINTS - 1))}

$PYTHON "$REPO/scripts/compute_uuas_by_relation.py" \
  --results-dir "$RESULTS_ROOT" --layers "$LAYERS"
