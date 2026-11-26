#!/bin/bash
# One structural probe at BERT-base checkpoint 16 (the paper's post-LayerNorm checkpoints, -convB files) per re-split, then
# its per-relation UASL on that split's dev and test sections.
#
#   OUT=/projects/.../ablation-splits-ck16 sbatch -A <acct> --array=0-19 run_split.sh
#
# Hyper-parameters are the published run's (experiments/bert-base-prd/results-convB/
# layer-16/BERTBase_convB_layer16.yaml); only corpus, embeddings, alignments and the
# seed change. Seed = split index + 1 (seed 0 is a no-op in run_experiment.py).
#SBATCH --partition=short
#SBATCH --job-name=split-ck16
#SBATCH --output=slurm-logs/split-ck16-%A_%a.log
set -euo pipefail
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[ -f "$REPO/paths.sh" ] || { echo "submit from the repo root" >&2; exit 1; }
source "$REPO/paths.sh"
export PYTHONPATH="$REPO/structural-probes"
: "${OUT:?set OUT}"

LAYER=16
KEY=$(printf split_%02d "$SLURM_ARRAY_TASK_ID")
SEED=$((SLURM_ARRAY_TASK_ID + 1))
RES="$OUT/results/$KEY/layer-$LAYER"
mkdir -p "$RES"
if [ -s "$RES/dev.uuas_by_relation" ] && [ -s "$RES/test.uuas_by_relation" ]; then
  echo "$KEY done"; exit 0; fi

CFG="$RES/${KEY}_layer$LAYER.yaml"
$PYTHON - <<PY
import yaml
cfg = yaml.safe_load(open("$REPO/experiments/bert-base-prd/results-convB/layer-16/BERTBase_convB_layer16.yaml"))
c, e, m = cfg['dataset']['corpus'], cfg['dataset']['embeddings'], cfg['model']
c['root'] = "$OUT/data/$KEY/"
c.update(train_path='train.conllx', dev_path='dev.conllx', test_path='test.conllx')
e['root'] = "$OUT/embeddings/$KEY/"
e.update(train_path='train.hdf5', dev_path='dev.hdf5', test_path='test.hdf5')
m['alignment_root'] = "$OUT/alignments/$KEY/"
m.update(alignment_train_path='train.hdf5', alignment_dev_path='dev.hdf5',
         alignment_test_path='test.hdf5', model_layer=$LAYER)
cfg['reporting']['root'] = "$RES"
yaml.safe_dump(cfg, open("$CFG", 'w'))
PY

echo "=== $KEY  layer $LAYER  seed $SEED ==="
$PYTHON "$REPO/structural-probes/run_experiment.py" "$CFG" \
  --results-dir "$RES" --train-probe 1 --report-results 1 --seed "$SEED"
$PYTHON "$REPO/scripts/compute_uuas_by_relation.py" --results-dir "$OUT/results/$KEY" --layers $LAYER
echo "$KEY dev.uuas $(cat "$RES/dev.uuas")"
