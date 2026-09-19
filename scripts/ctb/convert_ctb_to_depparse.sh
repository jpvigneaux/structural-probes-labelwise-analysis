#!/bin/bash
# Chinese counterpart of scripts/convert_splits_to_depparse.sh: convert the bare
# CTB trees written by scripts/ctb/extract_trees.py to basic Stanford
# Dependencies (CoNLL-X), with the same CoreNLP release and the same flags as the
# English conversion (-basic -keepPunct -conllx -checkConnected).
#
#   CORENLP=/path/to/stanford-corenlp-4.5.6 TREES=/path/to/trees OUT=/path/to/corpus \
#     bash scripts/ctb/convert_ctb_to_depparse.sh dev test
#
# Java must be on PATH (e.g. `module load java/jdk-17.0.2+8`).
set -euo pipefail
: "${CORENLP:?set CORENLP to the stanford-corenlp-4.5.6 directory}"
: "${TREES:?set TREES to the directory holding ctb9-nmw-<split>.trees}"
: "${OUT:?set OUT to the corpus directory}"
STEM=${STEM:-ctb9-nmw}
for split in "${@:-train dev test}"; do
  java -cp "$CORENLP/*" -mx1g \
    edu.stanford.nlp.trees.international.pennchinese.ChineseGrammaticalStructure \
    -treeFile "$TREES/$STEM-$split.trees" -basic -keepPunct -conllx -checkConnected \
    > "$OUT/$STEM-$split.conllx"
  n_trees=$(wc -l < "$TREES/$STEM-$split.trees")
  n_sents=$(awk 'BEGIN{n=0} /^$/{n++} END{print n}' "$OUT/$STEM-$split.conllx")
  echo "$split: $n_trees trees -> $n_sents CoNLL-X sentences"
  [ "$n_trees" -eq "$n_sents" ] || { echo "sentence count mismatch in $split" >&2; exit 1; }
done
