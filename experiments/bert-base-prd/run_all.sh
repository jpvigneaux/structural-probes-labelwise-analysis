#!/bin/bash
#SBATCH --account=p33044  ## YOUR ACCOUNT pXXXX or bXXXX
#SBATCH --partition=gengpu  ### PARTITION (buyin, short, normal, etc)
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 ## how many computers do you need
#SBATCH --ntasks-per-node=1 ## how many cpus or processors do you need on each computer
#SBATCH --time=04:00:00 ## how long does this need to run (remember different partitions have restrictions on this parameter)
#SBATCH --mem=5G ## how much RAM do you need per node (this effects your FairShare score so be careful to not ask for more than you need))
#SBATCH --job-name=train-bert-base-probes  ## When you run squeue -u NETID this is how you can identify the job
#SBATCH --output=slurm-logs/%x-%j.log  ## submit from experiments/bert-base-prd/

source "/path/to/structural-probes-labelwise-analysis/paths.sh"

source "$CONDA_INIT"
conda activate "$CONDA_ENV"

EXPERIMENT_DIR="$REPO_ROOT/experiments/bert-base-prd"
BASE_CONFIG="$EXPERIMENT_DIR/base_config.yaml"
RESULTS_DIR="$EXPERIMENT_DIR/results"
PROBE_SCRIPT="$REPO_ROOT/structural-probes/run_experiment.py"

export PYTHONPATH="$REPO_ROOT/structural-probes"

for LAYER in $(seq 9 12); do
    LAYER_PAD=$(printf "%02d" "$LAYER")
    LAYER_RESULTS="$RESULTS_DIR/layer-$LAYER_PAD"
    LAYER_CONFIG="/tmp/probe_bertbase_prd_layer${LAYER_PAD}.yaml"

    echo "=== Layer $LAYER ==="

    # Write a per-layer config by patching model_layer in the base config
    python3 - <<PYEOF
import yaml
with open("$BASE_CONFIG") as f:
    cfg = yaml.safe_load(f)
cfg['model']['model_layer'] = $LAYER
cfg['dataset']['corpus']['root'] = "$DATA_DIR"
cfg['dataset']['embeddings']['root'] = "$EMB_DIR_BERT_BASE"
with open("$LAYER_CONFIG", 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
PYEOF

    mkdir -p "$LAYER_RESULTS"

    python3 "$PROBE_SCRIPT" "$LAYER_CONFIG" \
        --results-dir "$LAYER_RESULTS" \
        --train-probe 1 \
        --report-results 1

    echo "--- Layer $LAYER done. Results in $LAYER_RESULTS ---"
done

echo ""
echo "All layers complete. Run plot_results.py to generate the figure."
