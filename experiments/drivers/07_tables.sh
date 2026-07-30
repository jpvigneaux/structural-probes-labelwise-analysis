#!/bin/bash
# Every regression the paper reports, for one run, at its optimal checkpoint.
#
#   RUN=bertbase RESULTS_ROOT=... sbatch experiments/drivers/07_tables.sh
#
# Produces, in order: Table 1 / the cross-model table row; the moment ladder;
# the standard-deviation-versus-variance comparison; and mean(log n) versus
# log(mean n). Run 09_verify.sh afterwards to diff the lot against the paper.
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --job-name=tables
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --output=slurm-logs/tables-%x-%j.log
# Locate the repository. sbatch copies this script into the SLURM spool
# directory before running it, so its own path says nothing about where the repo
# is: prefer REPO_ROOT (exported by submit_all.sh), then the directory sbatch was
# invoked from, and fall back to this file's location for a direct shell run.
REPO="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
[ -n "$REPO" ] && [ -f "$REPO/paths.sh" ] || \
  REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)" || true
[ -f "$REPO/paths.sh" ] || {
  echo "cannot locate the repository (no paths.sh found). Submit from the repo" >&2
  echo "root, or export REPO_ROOT=/path/to/structural-probes-labelwise-analysis." >&2
  exit 1; }
source "$REPO/experiments/drivers/common.sh"

: "${RESULTS_ROOT:=$(M "$RUN" results --path)}"
SIM=$(M --predictor sim)
LEN=$(M --predictor length)
U="$RESULTS_ROOT/layer-$(printf '%02d' "$OPT_CK")/dev.uuas_by_relation"
[ -s "$U" ] || { echo "missing $U -- run 05_ulas.sh first" >&2; exit 1; }

echo; echo "############ the published regression (Table 1 / cross-model row) ############"
$PYTHON "$REPO/scripts/regression/ptb_wls_regression.py" \
  -uuas "$U" -sim "$SIM" -len "$LEN" --label "$LABEL"

echo; echo "############ moment ladder: does dispersion or skew pay for itself? ############"
$PYTHON "$REPO/scripts/regression/add_length_spread.py" \
  -uuas "$U" -sim "$SIM" -len "$LEN" --label "$LABEL" --full

echo; echo "############ standard deviation vs variance ############"
$PYTHON "$REPO/scripts/regression/compare_dispersion_scale.py" \
  -uuas "$U" -sim "$SIM" -len "$LEN" --label "$LABEL"

echo; echo "############ mean(log n) vs log(mean n) ############"
$PYTHON "$REPO/scripts/regression/compare_length_predictors.py" \
  -uuas "$U" -sim "$SIM" -len "$LEN" --label "$LABEL"

echo; echo "############ how much of the ULAS spread is real, not sampling noise? ############"
$PYTHON "$REPO/scripts/regression/ulas_reliability.py" \
  --spec "$LABEL:$RESULTS_ROOT:$OPT_CK"
echo "=== done ==="
