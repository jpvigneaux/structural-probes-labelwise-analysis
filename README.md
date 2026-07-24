# Label-wise analysis of structural probes' performance

Code and saved results for the paper *"Label-wise analysis of structural
probes' performance."* The paper disaggregates the structural probe of
[Hewitt & Manning (2019)](https://nlp.stanford.edu/pubs/hewitt2019structural.pdf)
by dependency relation — computing a labeled attachment score (ULAS) per
relation — and analyses which factors (arc length, relation frequency,
similarity-corrected head entropy) predict how well each relation is
reconstructed.

This repository is a self-contained extract of a larger working repository: it
contains **only** the code, configuration, and saved outputs needed to
reproduce the figures and tables of the paper. The bundled
[`structural-probes/`](structural-probes/) library is the upstream
Hewitt & Manning codebase (see [`UPSTREAM_README.md`](UPSTREAM_README.md));
everything under [`experiments/`](experiments/) is our own.

---

## What reproduces what

The paper contains 8 figures and 2 tables. The table below maps each to the
script that produces it and the data it consumes.

| Paper artifact | Script | Input |
|---|---|---|
| Fig 1 — ULAS by relation (verb arguments) | [`experiments/bert-base-prd/figures/plot_selected_relations.py`](experiments/bert-base-prd/figures/plot_selected_relations.py) | `bert-base-prd/results-hface/layer-*/training_uuas_by_relation.tsv` |
| Fig 2 — ULAS by relation (top/mid/low) | [`experiments/bert-base-prd/figures/plot_selected_relations_2.py`](experiments/bert-base-prd/figures/plot_selected_relations_2.py) | same |
| Fig 3 — ULAS by relation, RoBERTa-Shuffle-N1 | [`experiments/roberta-shufflen1-prd/figures/plot_selected_relations.py`](experiments/roberta-shufflen1-prd/figures/plot_selected_relations.py) | `roberta-shufflen1-prd/results/layer-*/` |
| Fig 4 — mean ULAS vs arc length | [`experiments/bert-base-prd/uuas_mean_curves_by_checkpoint.py`](experiments/bert-base-prd/uuas_mean_curves_by_checkpoint.py) → [`uuas_mean_curves_png.py`](experiments/bert-base-prd/uuas_mean_curves_png.py) | `results-hface/` + PTB corpus |
| Fig 5 — R² heatmap, log-linear decay | [`experiments/bert-base-prd/regression_uas_vs_log_distance.py`](experiments/bert-base-prd/regression_uas_vs_log_distance.py) | `.npz` (below) |
| Fig 6 — relation dendrogram (ULAS-only) | [`experiments/bert-base-prd/cluster_relations_by_distance.py`](experiments/bert-base-prd/cluster_relations_by_distance.py) `--alpha 1.0` | `.npz` (below) |
| **Fig 7 — head sim-entropy vs ULAS** | **external — produced independently by a collaborator; not in this repo** | — |
| Fig 8 — four-panel dendrograms (appendix) | [`experiments/bert-base-prd/cluster_relations_by_distance.py`](experiments/bert-base-prd/cluster_relations_by_distance.py) | `.npz` (below) |
| **Table 1 — entropy WLS regression** | **external — produced independently by a collaborator; not in this repo** | — |
| Table 2 — 90th-percentile arc length (appendix) | [`experiments/bert-base-prd/inspect_distance_ranges.py`](experiments/bert-base-prd/inspect_distance_ranges.py) / `cluster_relations_by_distance.py` | `.npz` + PTB corpus |

> **Note on Fig 7 and Table 1 (entropy analysis).** The similarity-corrected
> entropy plot and the weighted-least-squares regression predicting ULAS were
> produced independently by a collaborator with a separate codebase, and are
> **not** included here. Reproducing them requires that external code plus
> fastText `wiki-news` word vectors.

The `.npz` referenced above is
`experiments/bert-base-prd/figures/uuas_mean_curves_by_checkpoint.npz`,
the per-relation ULAS-by-distance curves for all checkpoints. It is written by
`uuas_mean_curves_by_checkpoint.py` (which needs the PTB corpus) and is bundled
here, so the regression, dendrogram, and range analyses downstream of it
reproduce **without** any external data.

---

## Two reproduction levels

### 1. Figures from bundled results (no GPU, no external data)

The saved probe outputs (`results-hface/`, roberta `results/`) and the curve
`.npz` are included, so most figures regenerate directly:

```bash
PY=/path/to/conda/envs/sp-env/bin/python   # or your own sp-env python
cd experiments/bert-base-prd

$PY figures/plot_selected_relations.py            # Fig 1
$PY figures/plot_selected_relations_2.py          # Fig 2
$PY cluster_relations_by_distance.py              # Fig 8 (four-panel)
$PY cluster_relations_by_distance.py --alpha 1.0 --out figures/relation_clustering_1panel.png  # Fig 6
$PY regression_uas_vs_log_distance.py             # Fig 5

cd ../roberta-shufflen1-prd
$PY figures/plot_selected_relations.py            # Fig 3
```

### 2. Full reproduction from scratch (SLURM + external data)

Requires the external data described below. This full chain (extract → align →
train → per-relation ULAS) has been verified end-to-end on a small PTB subset;
see [Natural-sentence embeddings](#natural-sentence-embeddings-our-variation).
Broad pipeline:

0. **Pre-download models (once, on a node with internet).** Compute nodes have
   no outbound internet, so cache the HF models the extraction scripts need:
   ```bash
   HF_HOME=/gpfs/scratch/<user>/hf-cache python scripts/predownload_models.py
   ```
   Set the same `HF_HOME` in `paths.sh` (jobs run with `HF_HUB_OFFLINE=1`).
1. **Prep PTB.** Convert PTB-WSJ constituency trees to Stanford Dependencies
   (`scripts/convert_splits_to_depparse.sh`) and to whitespace-tokenized text
   (`scripts/convert_conll_to_raw.py`).
2. **Extract embeddings.**
   `experiments/bert-base-prd/submit_extract_natural_embeddings.sh`
   (BERT-base, 26 checkpoints) and
   `experiments/roberta-shufflen1-prd/submit_extract_embeddings.sh`
   (RoBERTa-Shuffle-N1).
3. **Precompute alignments** (BERT ↔ PTB):
   `experiments/bert-base-prd/submit_precompute_alignments.sh`.
4. **Train probes** (one parse-distance probe per checkpoint):
   `sbatch experiments/bert-base-prd/run_all_hface_prealigned.sh`,
   `python experiments/roberta-shufflen1-prd/submit_train_probes.py`.
5. **Per-relation ULAS:** `compute_uuas_by_relation_dev_test.py` in each
   experiment directory.
6. **Distance curves & figures** as in level 1.

---

## Environment

```bash
# Always use the explicit interpreter path (conda activate is unreliable on the cluster).
PYTHON=/path/to/conda/envs/sp-env/bin/python

# Key pins: transformers==4.38.2 (data.py breaks on v5), torch, scipy, plotly,
# simple_slurm, omegaconf, pyyaml.  See requirements.txt.
```

All non-trivial jobs run on SLURM (CPU: `--partition=short`; GPU:
`--partition=gengpu --gres=gpu:1`).

## Configuring paths

Machine-local paths live in [`paths.yaml`](paths.yaml) (read by Python) and
[`paths.sh`](paths.sh) (sourced by shell scripts). Edit both before running:

- `repo_root` — already set to this repository's location.
- `data.corpus` — PTB-WSJ CoNLL-X (`ptb3-wsj-{train,dev,test}.conllx`).
- `data.embeddings.*`, `data.models.*`, `data.alignment.*` — external
  embedding / model / alignment directories (not shipped; large).
- `HF_HOME` (in `paths.sh`) — HuggingFace cache the offline jobs read; point it
  at a directory you own and populate it with `scripts/predownload_models.py`.

The saved result configs under `results-hface/` and `results/` retain the
original absolute paths from the run that produced them, kept as provenance.

## Natural-sentence embeddings (our variation)

Unlike Hewitt & Manning, who feed raw PTB tokens to BERT, we embed a
**natural-language rendering** of each sentence:
`scripts/convert_raw_to_bert_natural_sentences.py` maps PTB escapes to surface
forms (`-LRB-`→`(`, `` `` ``/`''`→`"`) and restores natural spacing via
`data.natural_sentence`, so representations reflect how BERT was actually
pretrained. It records **26 checkpoints** per sentence (raw embedding, layer-00,
then post-attention and post-block residual streams for each of 12 blocks) using
`transformer_lens.HookedEncoder`. Subword vectors are pooled back to PTB tokens
with `data.hface_alignment_deptb`, a two-step character-level Levenshtein
alignment (subwords → natural string → PTB string), precomputed by
`scripts/precompute_alignments.py`.

## External data (not included)

Too large to bundle; needed only for full reproduction (level 2):

- **PTB-WSJ** dependency parses (CoNLL-X). Requires an unmodified PTB3 license.
- **BERT-base** natural-sentence embeddings (26-layer HDF5).
- **RoBERTa-Shuffle-N1** model + embeddings (Sinha et al., EMNLP 2021).
- **HF model cache** for `bert-base-cased` / `roberta-base` — compute nodes are
  offline, so run `scripts/predownload_models.py` first (level-2 step 0). Note
  `transformer_lens` resolves `bert-base-cased` to its canonical id
  `google-bert/bert-base-cased`; the script fetches both.

## Layout

```
structural-probes-label-analysis/
├── structural-probes/     # Bundled upstream Hewitt & Manning library
├── scripts/               # Embedding extraction + PTB prep
├── experiments/
│   ├── bert-base-prd/            # BERT-base probes, ULAS, distance/cluster analysis
│   │   ├── results-hface/        # Saved probe outputs (26 checkpoints)
│   │   ├── figures/              # Generators + rendered paper figures
│   │   ├── tables/               # Regression / range CSV+HTML
│   │   └── data/sentences.txt    # De-PTBified natural sentences (embedding input)
│   └── roberta-shufflen1-prd/    # RoBERTa-Shuffle-N1 probes (Fig 3)
├── paths.yaml / paths.sh  # Machine-local paths — edit these
├── requirements.txt
└── UPSTREAM_README.md     # Original structural-probes README
```
