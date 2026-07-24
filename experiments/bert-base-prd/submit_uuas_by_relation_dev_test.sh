#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --job-name=uuas-relation-dev-test
#SBATCH --output=/path/to/structural-probes-labelwise-analysis/experiments/bert-base-prd/slurm-logs/%x-%j.log

# Computes dev.uuas_by_relation and test.uuas_by_relation for all 26 layers.
# Loads only dev+test embeddings (skips 39k-sentence train) and handles
# GPU-saved probe params on a CPU-only node via map_location='cpu'.

source "/path/to/structural-probes-labelwise-analysis/paths.sh"

EXPERIMENT_DIR="$REPO_ROOT/experiments/bert-base-prd"
export PYTHONPATH="$REPO_ROOT/structural-probes"

$PYTHON "$EXPERIMENT_DIR/compute_uuas_by_relation_dev_test.py"

echo "Done."
