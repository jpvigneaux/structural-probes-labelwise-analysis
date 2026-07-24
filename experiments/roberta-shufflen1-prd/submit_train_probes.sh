#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --job-name=train-shufflen1-probes
#SBATCH --output=slurm-logs/%x-%j.log  ## submit from experiments/roberta-shufflen1-prd/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/paths.sh"
EXPERIMENT_DIR="$REPO_ROOT/experiments/roberta-shufflen1-prd"
BASE_CONFIG="$EXPERIMENT_DIR/base_config.yaml"
RESULTS_DIR="$EXPERIMENT_DIR/results"
PROBE_SCRIPT="$REPO_ROOT/structural-probes/run_experiment.py"

export PYTHONPATH="$REPO_ROOT/structural-probes"

# RoBERTa base: 13 layers total (0 = embedding layer, 1-12 = transformer layers)
for LAYER in $(seq 0 12); do
    LAYER_PAD=$(printf "%02d" "$LAYER")
    LAYER_RESULTS="$RESULTS_DIR/layer-$LAYER_PAD"
    LAYER_CONFIG="/tmp/probe_shufflen1_prd_layer${LAYER_PAD}.yaml"

    echo "=== Layer $LAYER ==="

    $PYTHON - <<PYEOF
import yaml
with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_layer'] = $LAYER
cfg['dataset']['corpus']['root'] = "$DATA_DIR"
cfg['dataset']['embeddings']['root'] = "$EMB_DIR_ROBERTA_SHUFFLEN1"
with open("$LAYER_CONFIG", 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF

    mkdir -p "$LAYER_RESULTS"

    $PYTHON "$PROBE_SCRIPT" "$LAYER_CONFIG" \
        --results-dir "$LAYER_RESULTS" \
        --train-probe 1 \
        --report-results 1

    echo "--- Layer $LAYER done. Results in $LAYER_RESULTS ---"
done

echo ""
echo "All layers complete."
