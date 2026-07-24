#!/bin/bash
#SBATCH --account=p33044
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --job-name=bert-natural-emb
#SBATCH --output=/path/to/structural-probes-labelwise-analysis/experiments/bert-base-prd/slurm-logs/%x-%j.log  ## submit from experiments/bert-base-prd/

source "/path/to/structural-probes-labelwise-analysis/paths.sh"

mkdir -p "$EMB_DIR_BERT_BASE_NATURAL"

for SPLIT in train dev test; do
    echo "=== Extracting $SPLIT ==="
    $PYTHON "$REPO_ROOT/scripts/convert_raw_to_bert_natural_sentences.py" \
        "$DATA_DIR/ptb3-wsj-${SPLIT}.conllx" \
        "$EMB_DIR_BERT_BASE_NATURAL/raw.${SPLIT}.bertbase-natural-layers.hdf5" \
        base
    echo "--- $SPLIT done ---"
done

echo "All splits done."
