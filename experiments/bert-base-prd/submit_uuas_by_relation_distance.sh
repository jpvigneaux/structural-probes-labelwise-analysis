#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --job-name=uuas-relation-distance
#SBATCH --output=experiments/bert-base-prd/slurm-logs/%x-%j.log  ## submit from experiments/bert-base-prd/

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/paths.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/bert-base-prd"
cd $EXPERIMENT_DIR
$PYTHON uuas_by_relation_distance.py 

echo "All splits done."