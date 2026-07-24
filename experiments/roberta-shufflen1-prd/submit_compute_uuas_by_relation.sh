#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --job-name=shufflen1-uuas-by-rel
#SBATCH --output=/path/to/structural-probes-labelwise-analysis/experiments/roberta-shufflen1-prd/slurm-logs/%x-%j.log

source /path/to/structural-probes-labelwise-analysis/paths.sh

EXPERIMENT_DIR=/path/to/structural-probes-labelwise-analysis/experiments/roberta-shufflen1-prd

$PYTHON $EXPERIMENT_DIR/compute_uuas_by_relation_dev_test.py
