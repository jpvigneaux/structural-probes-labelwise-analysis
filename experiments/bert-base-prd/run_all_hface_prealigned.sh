#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=8G
#SBATCH --job-name=train-bert-base-probes-prealigned
#SBATCH --output=slurm-logs/%x-%j.log  ## submit from experiments/bert-base-prd/

# Train parse-distance probes using pre-aligned BERT-base natural-sentence
# embeddings.  Alignment matrices were pre-computed by
# submit_precompute_alignments.sh, so each probe invocation skips Levenshtein
# and does only a cheap matmul instead.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/paths.sh"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_DIR="$REPO_ROOT/experiments/bert-base-prd"
BASE_CONFIG="$EXPERIMENT_DIR/base_config_hface_prealigned.yaml"
RESULTS_DIR="$EXPERIMENT_DIR/results-hface"
PROBE_SCRIPT="$REPO_ROOT/structural-probes/run_experiment.py"

export PYTHONPATH="$REPO_ROOT/structural-probes"

for IDX in $(seq 0 25); do
    LAYER_PAD=$(printf "%02d" "$IDX")
    LAYER_RESULTS="$RESULTS_DIR/layer-$LAYER_PAD"
    LAYER_CONFIG="/tmp/probe_bertbase_prealigned_layer${LAYER_PAD}.yaml"

    echo "=== Layer $LAYER_PAD (HDF5 index $IDX) ==="

    $PYTHON - <<PYEOF
import yaml
with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_layer'] = $IDX
cfg['dataset']['corpus']['root'] = "$DATA_DIR"
cfg['dataset']['embeddings']['root'] = "$EMB_DIR_BERT_BASE_NATURAL"
cfg['model']['alignment_root'] = "$ALIGN_DIR_BERT_BASE_NATURAL"
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
