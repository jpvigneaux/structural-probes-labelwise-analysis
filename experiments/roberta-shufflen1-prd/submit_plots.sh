#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --job-name=plot-shufflen1
#SBATCH --output=slurm-logs/%x-%j.log  ## submit from experiments/roberta-shufflen1-prd/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/paths.sh"
EXPERIMENT_DIR="$REPO_ROOT/experiments/roberta-shufflen1-prd"
FIGURES_DIR="$EXPERIMENT_DIR/figures"
LOGS=$(ls $EXPERIMENT_DIR/slurm-logs/train-shufflen1-probes-*.log 2>/dev/null | tr '\n' ' ')

mkdir -p "$FIGURES_DIR"

cd "$REPO_ROOT"

# --- Plot 1: overall UUAS vs layer ---
$PYTHON experiments/plot_uuas_by_layer.py \
    --results-dir "$EXPERIMENT_DIR/results" \
    --output      "$FIGURES_DIR/uuas_by_layer.pdf" \
    --model-name  "RoBERTa-shuffle-N1"

# --- Plot 2: per-relation UUAS at optimal epoch vs layer (interactive HTML) ---
$PYTHON experiments/plot_optimal_uuas_by_relation_and_layer.py \
    --results-dir "$EXPERIMENT_DIR/results" \
    --logs        $LOGS \
    --output      "$FIGURES_DIR/optimal_uuas_by_relation_and_layer.html"

echo "Figures written to $FIGURES_DIR"
