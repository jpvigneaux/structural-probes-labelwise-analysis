#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --job-name=train-bert-base-probes-hface
#SBATCH --output=/path/to/structural-probes-labelwise-analysis/experiments/bert-base-prd/slurm-logs/%x-%j.log  ## submit from experiments/bert-base-prd/


# Train parse-distance probes on de-PTBified BERT-base embeddings with hface-deptb alignment.
#
# HDF5 index scheme (26 total):
#   0          emb        raw word embeddings
#   1          layer-00   embedding layer output
#   2k         layer-k+A  post-attention intermediate of block k  (k=1..12)
#   2k+1       layer-k    full output of block k                  (k=1..12)
#
# Each probe writes results to results-hface/layer-XX/ where XX = HDF5 index (00..25).

source "/path/to/structural-probes-labelwise-analysis/paths.sh"

EXPERIMENT_DIR="$REPO_ROOT/experiments/bert-base-prd"
BASE_CONFIG="$EXPERIMENT_DIR/base_config_hface.yaml"
RESULTS_DIR="$EXPERIMENT_DIR/results-hface"
PROBE_SCRIPT="$REPO_ROOT/structural-probes/run_experiment.py"

export PYTHONPATH="$REPO_ROOT/structural-probes"

for IDX in $(seq 0 25); do
    LAYER_PAD=$(printf "%02d" "$IDX")
    LAYER_RESULTS="$RESULTS_DIR/layer-$LAYER_PAD"
    LAYER_CONFIG="/tmp/probe_bertbase_hface_layer${LAYER_PAD}.yaml"

    echo "=== Layer $LAYER_PAD (HDF5 index $IDX) ==="

    $PYTHON - <<PYEOF
import yaml
with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_layer'] = $IDX
cfg['dataset']['corpus']['root'] = "$DATA_DIR"
cfg['dataset']['embeddings']['root'] = "$EMB_DIR_BERT_BASE_NATURAL"
with open("$LAYER_CONFIG", 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF

    mkdir -p "$LAYER_RESULTS"

    $PYTHON "$PROBE_SCRIPT" "$LAYER_CONFIG" \
        --results-dir "$LAYER_RESULTS" \
        --train-probe 1 \
        --report-results 1

    echo "--- Layer $LAYER_PAD done. Results in $LAYER_RESULTS ---"
done

echo ""
echo "All layers complete."
