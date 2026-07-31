#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --job-name=bert-natural-alignments
#SBATCH --output=experiments/bert-base-prd/slurm-logs/%x-%j.log  ## submit from experiments/bert-base-prd/

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/paths.sh"

mkdir -p "$ALIGN_DIR_BERT_BASE_NATURAL"

for SPLIT in train dev test; do
    echo "=== Computing alignments for $SPLIT ==="
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    $PYTHON "$REPO_ROOT/scripts/precompute_alignments.py" \
        "$DATA_DIR/ptb3-wsj-${SPLIT}.conllx" \
        "$ALIGN_DIR_BERT_BASE_NATURAL/alignments.${SPLIT}.hdf5" \
        base
    echo "--- $SPLIT done ---"
done

echo "All splits done."
