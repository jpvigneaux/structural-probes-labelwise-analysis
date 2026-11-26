#!/bin/bash
# One structural probe at BERT-base checkpoint 16 (the paper's post-LayerNorm checkpoints, -convB files) per random seed, on the
# published PTB split, then its per-relation UASL on dev and test. Only the seed differs
# from the published run (experiments/bert-base-prd/results-convB/layer-16/).
#
#   OUT=/projects/.../seeds-ck16 sbatch -A <acct> --array=0-19 run_seed.sh
#
# Seed = task index + 1 (seed 0 is a no-op in run_experiment.py).
#SBATCH --partition=short
#SBATCH --job-name=seed-ck16
#SBATCH --output=slurm-logs/seed-ck16-%A_%a.log
set -euo pipefail
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[ -f "$REPO/paths.sh" ] || { echo "submit from the repo root" >&2; exit 1; }
source "$REPO/paths.sh"
export PYTHONPATH="$REPO/structural-probes"
: "${OUT:?set OUT}"

LAYER=16
SEED=$((SLURM_ARRAY_TASK_ID + 1))
KEY=$(printf seed_%02d "$SEED")
RES="$OUT/results/$KEY/layer-$LAYER"
mkdir -p "$RES"
if [ -s "$RES/dev.uuas_by_relation" ] && [ -s "$RES/test.uuas_by_relation" ]; then
  echo "$KEY done"; exit 0; fi

# Corpus, embeddings and alignments of the published run, from paths.sh / paths.yaml.
M () { $PYTHON "$REPO/experiments/drivers/manifest.py" "$@"; }
EMB_DIR="$(M --root embeddings)/$(M bertbase extraction.embeddings)/"
ALIGN_DIR="$(M --root alignments)/$(M bertbase extraction.alignments)/"

CFG="$RES/${KEY}_layer$LAYER.yaml"
$PYTHON - <<PY
import yaml
cfg = yaml.safe_load(open("$REPO/experiments/bert-base-prd/results-convB/layer-16/BERTBase_convB_layer16.yaml"))
cfg['dataset']['corpus']['root'] = "$DATA_DIR"
cfg['dataset']['embeddings']['root'] = "$EMB_DIR"
cfg['model']['alignment_root'] = "$ALIGN_DIR"
cfg['model']['model_layer'] = $LAYER
cfg['reporting']['root'] = "$RES"
yaml.safe_dump(cfg, open("$CFG", 'w'))
PY

echo "=== $KEY  layer $LAYER ==="
$PYTHON "$REPO/structural-probes/run_experiment.py" "$CFG" \
  --results-dir "$RES" --train-probe 1 --report-results 1 --seed "$SEED"
$PYTHON "$REPO/scripts/compute_uuas_by_relation.py" --results-dir "$OUT/results/$KEY" --layers $LAYER
echo "$KEY dev.uuas $(cat "$RES/dev.uuas")"
