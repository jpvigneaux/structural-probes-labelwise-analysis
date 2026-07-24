#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --job-name=extract-shufflen1
#SBATCH --output=slurm-logs/%x-%j.log  ## submit from experiments/roberta-shufflen1-prd/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/paths.sh"

MODEL_DIR="$MODEL_DIR_ROBERTA_SHUFFLEN1"
EMB_DIR="$EMB_DIR_ROBERTA_SHUFFLEN1"
SCRIPT="$REPO_ROOT/scripts/convert_raw_to_roberta_shufflen1.py"

mkdir -p "$MODEL_DIR" "$EMB_DIR"

# --- Download model (skip if already present) ---
if [ ! -f "$MODEL_DIR/model.pt" ]; then
    echo "Downloading roberta.base.shuffle.n1 ..."
    cd "$MODEL_DIR"
    wget -q https://dl.fbaipublicfiles.com/unnatural_pretraining/roberta.base.shuffle.n1.tar.gz
    tar -xzf roberta.base.shuffle.n1.tar.gz --strip-components=1
    rm roberta.base.shuffle.n1.tar.gz
    echo "Model downloaded to $MODEL_DIR"
else
    echo "Model already present at $MODEL_DIR/model.pt"
fi

cd "$REPO_ROOT"

# --- Extract embeddings for each split ---
for SPLIT in train dev test; do
    OUTPUT="$EMB_DIR/raw.${SPLIT}.roberta-shufflen1-layers.hdf5"
    if [ -f "$OUTPUT" ]; then
        echo "Skipping $SPLIT: $OUTPUT already exists."
        continue
    fi
    echo "=== Extracting $SPLIT ==="
    $PYTHON "$SCRIPT" \
        --model-dir "$MODEL_DIR" \
        --input     "$DATA_DIR/ptb3-wsj-${SPLIT}-raw.txt" \
        --output    "$OUTPUT"
done

echo ""
echo "All splits done. Embeddings in $EMB_DIR"
