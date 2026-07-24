#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --job-name=uuas-mean-curves
#SBATCH --output=/path/to/structural-probes-labelwise-analysis/experiments/bert-base-prd/slurm-logs/%x-%j.log  ## submit from experiments/bert-base-prd/

source "/path/to/structural-probes-labelwise-analysis/paths.sh"

EXPERIMENT_DIR="$REPO_ROOT/experiments/bert-base-prd"
cd "$EXPERIMENT_DIR"

$PYTHON uuas_mean_curves_by_checkpoint.py \
    --trained-probes-dir results-hface \
    --checkpoints 3 5 7 9 11 13 15 16 17 19 21 23 25 \
    --out figures/uuas_mean_curves_by_checkpoint.html

echo "Done."
