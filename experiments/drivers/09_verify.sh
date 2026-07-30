#!/bin/bash
# Diff every number in the paper against what the code produces now.
#
#   sbatch experiments/drivers/09_verify.sh
#
# Exits non-zero only on a genuine mismatch. Values that differ by one unit in
# the last printed digit because the paper rounded a 4-decimal printout again to
# 3 are reported separately and do not fail the run.
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --job-name=verify
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm-logs/verify-%j.log
set -uo pipefail
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
source "$REPO/paths.sh"

echo "############ embeddings and alignments ############"
for RUN in $($PYTHON "$REPO/experiments/drivers/manifest.py" --keys); do
  $PYTHON "$REPO/scripts/check_embeddings.py" --run "$RUN" --splits dev || true
done

echo; echo "############ published numbers ############"
$PYTHON "$REPO/scripts/verify_paper_numbers.py"
