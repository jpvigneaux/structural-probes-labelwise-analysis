#!/bin/bash
# Shared setup for every driver in this directory. Not runnable on its own.
#
# Defines: REPO, PYTHON, DRV, M (manifest query), and the roots from
# experiments/paper_runs.yaml. Each driver takes the run key in $RUN.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/paths.sh"
export PYTHONPATH="$REPO/structural-probes"
DRV="$REPO/experiments/drivers"

# M <run> <field> [--path]  ->  one value from the manifest
M () { $PYTHON "$DRV/manifest.py" "$@"; }

: "${RUN:?set RUN to a key from experiments/paper_runs.yaml (see: manifest.py --keys)}"

LABEL=$(M "$RUN" label)
HF_NAME=$(M "$RUN" extraction.hf_name)
TOKENIZER=$(M "$RUN" extraction.tokenizer)
LAYOUT=$(M "$RUN" extraction.layout)
CONVENTION=$(M "$RUN" extraction.layernorm_convention)
STEM=$(M "$RUN" extraction.stem)
HIDDEN=$(M "$RUN" extraction.hidden_dim)
EMB_NAME=$(M "$RUN" extraction.embeddings)
ALIGN_NAME=$(M "$RUN" extraction.alignments)
REVISION=$(M "$RUN" extraction.revision --default '')
N_CHECKPOINTS=$(M "$RUN" extraction.n_checkpoints)
OPT_CK=$(M "$RUN" optimal_checkpoint)
POST_BLOCK=$(M "$RUN" post_block_checkpoints)

CORPUS=$(M --root corpus)
EMB_ROOT=$(M --root embeddings)
ALIGN_ROOT=$(M --root alignments)
OUT=$(M --root out)

EMB_DIR="$EMB_ROOT/$EMB_NAME"
ALIGN_DIR="$ALIGN_ROOT/$ALIGN_NAME"

echo "=== $LABEL  ($RUN) ==="
echo "    model        $HF_NAME${REVISION:+  (revision $REVISION)}"
echo "    layout       $LAYOUT, $N_CHECKPOINTS checkpoints, LayerNorm convention $CONVENTION"
echo "    embeddings   $EMB_DIR"
echo "    alignments   $ALIGN_DIR"
