#!/bin/bash
# Extract residual-stream checkpoints for one model, all three splits.
#
#   RUN=gpt2 sbatch experiments/drivers/01_extract.sh
#
# Which extractor runs is read from the manifest, not chosen here: four of the
# runs use the 2+2L extractor and two use the 1+L one, and the choice renumbers
# every checkpoint downstream.
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=extract
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=slurm-logs/extract-%x-%j.log
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

EXTRACTOR=$(M "$RUN" extraction.extractor)
STORE_DTYPE=$(M "$RUN" extraction.store_dtype --default float32)
mkdir -p "$EMB_DIR"

for SPLIT in train dev test; do
  OUT_H5="$EMB_DIR/raw.$SPLIT.$STEM.hdf5"
  if [ -s "$OUT_H5" ]; then echo "$SPLIT already extracted, skipping"; continue; fi

  case "$EXTRACTOR" in
    convert_raw_to_hf_all_checkpoints.py*)
      # --allow-missing-midblock is part of the manifest's extractor string for
      # GPT-J, whose parallel residual branch has no post-attention state.
      EXTRA=""
      [[ "$EXTRACTOR" == *allow-missing-midblock* ]] && EXTRA="--allow-missing-midblock"
      $PYTHON "$REPO/scripts/convert_raw_to_hf_all_checkpoints.py" \
        "$CORPUS/ptb3-wsj-$SPLIT.conllx" "$OUT_H5" "$HF_NAME" \
        --dtype "$STORE_DTYPE" --model-dtype "$STORE_DTYPE" \
        ${REVISION:+--revision "$REVISION"} $EXTRA
      ;;
    convert_raw_to_hf_natural_sentences.py)
      $PYTHON "$REPO/scripts/convert_raw_to_hf_natural_sentences.py" \
        "$CORPUS/ptb3-wsj-$SPLIT.conllx" "$OUT_H5" "$HF_NAME" \
        --dtype "$STORE_DTYPE" ${REVISION:+--revision "$REVISION"}
      ;;
    convert_raw_to_roberta_shufflen1.py)
      # Fairseq checkpoint converted to HF weights on the fly; see the script.
      $PYTHON "$REPO/scripts/convert_raw_to_roberta_shufflen1.py" \
        "$CORPUS/ptb3-wsj-$SPLIT.conllx" "$OUT_H5" \
        --model-dir "$(M --root models)/roberta.base.shuffle.n1" \
        --dtype "$STORE_DTYPE"
      ;;
    *) echo "unknown extractor '$EXTRACTOR' for run $RUN" >&2; exit 1 ;;
  esac
done

echo "=== checking the written files against the manifest ==="
$PYTHON "$REPO/scripts/check_embeddings.py" --run "$RUN"
