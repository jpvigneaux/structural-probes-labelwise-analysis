#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --job-name=shufflen1-uuas-by-rel
#SBATCH --output=experiments/roberta-shufflen1-prd/slurm-logs/%x-%j.log

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source $REPO_ROOT/paths.sh

EXPERIMENT_DIR=$REPO_ROOT/experiments/roberta-shufflen1-prd

$PYTHON $EXPERIMENT_DIR/compute_uuas_by_relation_dev_test.py
