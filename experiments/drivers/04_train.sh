#!/bin/bash
# Train one structural probe per checkpoint. Array index = checkpoint index.
#
#   RUN=gpt2 sbatch --array=0-25 experiments/drivers/04_train.sh
#
# The array range must cover the run's checkpoint count:
#   manifest.py $RUN extraction.n_checkpoints  ->  N,  so --array=0-$((N-1))
# submit_all.sh works this out for you.
#
# Account p33044 is capped at 8 concurrent GPU jobs, so a long pending queue is
# expected and is not a scheduling fault. Add %6 to the array spec to stay well
# inside the cap when other work is running.
#SBATCH --account=p33044
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=probe
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm-logs/probe-%x-%A_%a.log
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

: "${SLURM_ARRAY_TASK_ID:?this script must be submitted as an array job}"
: "${RESULTS_ROOT:?set RESULTS_ROOT to the directory that will hold layer-NN/}"

L=$SLURM_ARRAY_TASK_ID
PAD=$(printf %02d "$L")
RES="$RESULTS_ROOT/layer-$PAD"
CFG_DIR="$RESULTS_ROOT/configs"
CFG="$CFG_DIR/${RUN}_layer$PAD.yaml"
mkdir -p "$RES" "$CFG_DIR"

if [ -s "$RES/predictor.params" ] && [ -s "$RES/dev.uuas" ]; then
  echo "$RUN checkpoint $PAD already complete, skipping"; exit 0
fi

$PYTHON "$REPO/scripts/make_probe_config.py" --run "$RUN" --layer "$L" \
  --embeddings-dir "$EMB_DIR" --alignments-dir "$ALIGN_DIR" \
  --corpus "$CORPUS" --out "$CFG"

$PYTHON "$REPO/structural-probes/run_experiment.py" "$CFG" \
  --results-dir "$RES" --train-probe 1 --report-results 1

echo "$RUN checkpoint $PAD dev.uuas: $(cat "$RES/dev.uuas" 2>/dev/null)"
