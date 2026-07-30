#!/bin/bash
# Every figure the paper draws for one run.
#
#   RUN=modernbert RESULTS_ROOT=... FIGS=... sbatch experiments/drivers/08_figures.sh
#
# The checkpoint set passed to the curve builder is the run's post-block series,
# read from the manifest: `3 5 ... 2L+1` for a 2+2L layout and `1 ... L` for a
# 1+L one. Passing the wrong one silently mixes post-attention and post-block
# states into a single series.
#SBATCH --account=p33044
#SBATCH --partition=short
#SBATCH --job-name=figures
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=slurm-logs/figures-%x-%j.log
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
: "${FIGS:=$(M --root figs)}"
S="$REPO/scripts/figures"
mkdir -p "$FIGS"

REF_CK=$(M "$RUN" best_post_block)
if [ "$REF_CK" = "$OPT_CK" ]; then
  REF_DESC="Optimal checkpoint"
else
  # For a 2+2L model the overall optimum can be a post-attention index that the
  # post-block series excludes, so the figure shows the best post-block one and
  # says so rather than quietly relabelling it "optimal".
  REF_DESC="Best post-block checkpoint"
fi

echo "=== per-relation ULAS across checkpoints ==="
$PYTHON "$REPO/scripts/plot_selected_relations.py" --results-dir "$RESULTS_ROOT" \
  --out "$FIGS/selected_uuas_by_relation_$RUN.png" --model-label "$LABEL"

echo "=== ULAS-by-arc-length curves (checkpoints: $POST_BLOCK) ==="
$PYTHON "$S/uuas_mean_curves_by_checkpoint.py" \
  --trained-probes-dir "$RESULTS_ROOT" --checkpoints $POST_BLOCK \
  --out "$FIGS/uuas_mean_curves_$RUN.html"
NPZ="$FIGS/uuas_mean_curves_$RUN.npz"

echo "=== two-panel curve figure (reference checkpoint $REF_CK) ==="
$PYTHON "$S/uuas_mean_curves_png.py" --curves "$NPZ" \
  --out "$FIGS/uuas_mean_curves_png_$RUN.png" \
  --ref-checkpoint "$REF_CK" --model-label "$LABEL" --ref-desc "$REF_DESC"

echo "=== goodness of fit of the log-linear decay model ==="
# Writes to a fixed path derived from the curves file, so each model would
# otherwise overwrite the last; move the output under the run's own name.
$PYTHON "$S/regression_uas_vs_log_distance.py" --curves "$NPZ"
mv -f "$FIGS/../figures/r2_heatmap_uas_vs_log_distance.png" "$FIGS/r2_heatmap_$RUN.png"
mv -f "$FIGS/../tables/regression_uas_vs_log_distance.csv" "$FIGS/r2_table_$RUN.csv"

echo "=== dendrogram, alpha = 1 (compact, for the tiled appendix figure) ==="
# Generated at the size it is printed at (0.48 x 6.30in textwidth), so the
# stated font sizes are the ones the reader gets.
$PYTHON "$S/cluster_relations_by_distance.py" --curves "$NPZ" --alpha 1.0 \
  --no-title --figsize 3.02,4.2 --leaf-font 5.0 --label-font 7.0 --dpi 400 \
  --out "$FIGS/relation_clustering_1panel_compact_$RUN.png"

echo "=== done ==="
ls -la "$FIGS" | grep -- "$RUN" || true
