#!/bin/bash
# Submit the whole pipeline for one run, with the SLURM dependencies wired up.
#
#   experiments/drivers/submit_all.sh gpt2 /scratch/$USER/probes/gpt2
#
# Run this from a login node -- it only submits. The array bounds are derived
# from the manifest, so a model with 46 checkpoints gets 46 tasks and one with
# 13 gets 13; hardcoding a range here is how probes end up trained on
# checkpoints that do not exist.
#
# The GPU stages (extract, train) are the expensive ones. The account is
# capped at 8 concurrent GPU jobs, so the training array is throttled to 6 to
# leave headroom for other work; a long pending queue is expected.
#
# 06_predictors.sh is deliberately NOT submitted here: the corpus statistics are
# identical for every model and want computing once. Run it separately, first.
set -euo pipefail

RUN=${1:?usage: submit_all.sh RUN RESULTS_ROOT}
RESULTS_ROOT=${2:?usage: submit_all.sh RUN RESULTS_ROOT}

DRV="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$DRV/../.." && pwd)"
source "$REPO/paths.sh"
mkdir -p "$REPO/slurm-logs" "$RESULTS_ROOT"

N=$($PYTHON "$DRV/manifest.py" "$RUN" extraction.n_checkpoints)
LAST=$((N - 1))
CONV=$($PYTHON "$DRV/manifest.py" "$RUN" extraction.layernorm_convention)
echo "run $RUN: $N checkpoints, LayerNorm convention $CONV -> $RESULTS_ROOT"

# sbatch propagates the environment, and the stage scripts are copied into the
# SLURM spool directory before they run, so REPO_ROOT is how they find the repo.
REPO_ROOT="$REPO"
export RUN RESULTS_ROOT REPO_ROOT

j_extract=$(sbatch --parsable --job-name="ex-$RUN" "$DRV/01_extract.sh")
echo "  01 extract      $j_extract"

dep_align=""
if [ "$CONV" = "B" ]; then
  j_conv=$(sbatch --parsable --dependency=afterok:$j_extract \
           --job-name="cv-$RUN" "$DRV/02_convention.sh")
  echo "  02 convention   $j_conv  (after $j_extract)"
  dep_embed=$j_conv
else
  echo "  02 convention   skipped (convention $CONV needs no transform)"
  dep_embed=$j_extract
fi

# Alignments depend only on the tokenizer, so they run alongside extraction.
j_align=$(sbatch --parsable --job-name="al-$RUN" "$DRV/03_align.sh")
echo "  03 align        $j_align  (independent of extraction)"

j_train=$(sbatch --parsable --array=0-$LAST%6 \
          --dependency=afterok:$dep_embed:$j_align \
          --job-name="pr-$RUN" "$DRV/04_train.sh")
echo "  04 train        $j_train  (array 0-$LAST%6, after $dep_embed and $j_align)"

j_ulas=$(sbatch --parsable --array=0-$LAST --dependency=afterok:$j_train \
         --job-name="ul-$RUN" "$DRV/05_ulas.sh")
echo "  05 UASL         $j_ulas  (array 0-$LAST, after $j_train)"

j_tab=$(sbatch --parsable --dependency=afterok:$j_ulas \
        --job-name="tb-$RUN" "$DRV/07_tables.sh")
echo "  07 tables       $j_tab  (after $j_ulas)"

j_fig=$(sbatch --parsable --dependency=afterok:$j_ulas \
        --job-name="fg-$RUN" "$DRV/08_figures.sh")
echo "  08 figures      $j_fig  (after $j_ulas)"

echo
echo "When every run has finished, point roots: in experiments/paper_runs.yaml at"
echo "your output and submit experiments/drivers/09_verify.sh."
