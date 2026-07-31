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

Everything in the paper is now produced by this repository, including the
weighted-least-squares regression and the similarity-corrected entropy that
earlier versions of this README described as external. Artifacts are named by
their LaTeX label rather than by number, since float placement moves the numbers
around.

Six probing runs stand behind the paper. Which model, checkpoint layout,
LayerNorm convention and tokenizer each uses is recorded in
[`experiments/paper_runs.yaml`](experiments/paper_runs.yaml), and the drivers in
[`experiments/drivers/`](experiments/drivers/) read it rather than hardcoding
anything.

### Main text

| Paper artifact | Script |
|---|---|
| `fig:dependencies1`, `fig:dependencies2` — ULAS by relation | [`scripts/plot_selected_relations.py`](scripts/plot_selected_relations.py) |
| `fig:dependencies-shuffled` — the same for RoBERTa-Shuffle-n1 | same, on the `shufflen1` run |
| `fig:performance-as-a-function-of-distance` — mean ULAS vs arc length | [`scripts/figures/uuas_mean_curves_by_checkpoint.py`](scripts/figures/uuas_mean_curves_by_checkpoint.py) → [`uuas_mean_curves_png.py`](scripts/figures/uuas_mean_curves_png.py) |
| `fig:R2-log-distance-model` — R² heat map, log-linear decay | [`scripts/figures/regression_uas_vs_log_distance.py`](scripts/figures/regression_uas_vs_log_distance.py) |
| `fig:ULAS-only-dendrogram` — relation dendrogram, α = 1 | [`scripts/figures/cluster_relations_by_distance.py`](scripts/figures/cluster_relations_by_distance.py) `--alpha 1.0` |
| `tab:regression_results` — the three-predictor WLS regression | [`scripts/regression/ptb_wls_regression.py`](scripts/regression/ptb_wls_regression.py) |

### Appendices

| Paper artifact | Script |
|---|---|
| `tab:arc-len-90` — 90th-percentile arc length | [`scripts/figures/cluster_relations_by_distance.py`](scripts/figures/cluster_relations_by_distance.py) |
| `fig:dendrogram_relations`, `fig:dendrogram_w1` — four-panel dendrograms over α | same, `--range-metric p90` / `w1` |
| `tab:other-models` — the regression, repeated on five models | [`scripts/regression/ptb_wls_regression.py`](scripts/regression/ptb_wls_regression.py) per run |
| `fig:app-selected`, `fig:app-curves`, `fig:app-r2a`, `fig:app-r2b`, `fig:app-dendro` | [`experiments/drivers/08_figures.sh`](experiments/drivers/08_figures.sh) per run |
| `fig:predictor-grid` — predictors against ULAS | [`scripts/plot_predictor_grid.py`](scripts/plot_predictor_grid.py) |
| `tab:moment-ladder` — does dispersion or skew pay for itself? | [`scripts/regression/add_length_spread.py`](scripts/regression/add_length_spread.py) |
| sd vs variance (Appendix "Standard deviation, not variance") | [`scripts/regression/compare_dispersion_scale.py`](scripts/regression/compare_dispersion_scale.py) |
| mean(log n) vs log(mean n) (same appendix) | [`scripts/regression/compare_length_predictors.py`](scripts/regression/compare_length_predictors.py) |
| `tab:shuffled-regression`, `fig:app-shuffled-curves`, `fig:app-shuffled-r2` | the same scripts, on the `shufflen1` run |
| the reliability ceiling (Appendix "not a floor effect") | [`scripts/regression/ulas_reliability.py`](scripts/regression/ulas_reliability.py) |
| the similarity-corrected entropy predictor | [`scripts/regression/contextual_sim_entropy.py`](scripts/regression/contextual_sim_entropy.py) |

### Checking that it reproduced

```bash
sbatch experiments/drivers/09_verify.sh
```

`scripts/check_embeddings.py` validates each run's embeddings and alignments
against the manifest; `scripts/verify_paper_numbers.py` refits every regression
and diffs it against the published value. On the archived runs all six pass the
first and all 88 published values reproduce exactly as printed. Five of them were
corrected in the paper after this check first ran — see
[REPRODUCING.md](REPRODUCING.md#0-the-manifest-and-how-to-check-you-have-reproduced-anything)
for which, and why they were wrong.

The `.npz` referenced by the figure scripts is
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

Requires the external data described below, and covers all six probing runs.
[REPRODUCING.md](REPRODUCING.md) is the detailed account; the short version:

0. **Pre-download models (once, on a node with internet).** Compute nodes have
   no outbound internet, so cache the HF models the extraction scripts need:
   ```bash
   HF_HOME=/gpfs/scratch/<user>/hf-cache python scripts/predownload_models.py
   ```
   Set the same `HF_HOME` in `paths.sh` (jobs run with `HF_HUB_OFFLINE=1`).
1. **Prep PTB.** Convert PTB-WSJ constituency trees to Stanford Dependencies
   (`scripts/convert_splits_to_depparse.sh`) and to whitespace-tokenized text
   (`scripts/convert_conll_to_raw.py`).
2. **Fill in `paths.yaml` and `paths.sh`** from the templates (see
   [Configuring paths](#configuring-paths)); `roots:` is where your output goes.
3. **Corpus statistics, once:** `sbatch experiments/drivers/06_predictors.sh`.
   These are identical for every model — only the ULAS values differ between
   rows of `tab:other-models`.
4. **Each run:** `experiments/drivers/submit_all.sh <run> <results-dir>`, which
   submits extraction → LayerNorm convention → alignment → probe training →
   per-relation ULAS → tables → figures with the SLURM dependencies wired up and
   the array bounds derived from the run's own checkpoint count.
   ```bash
   for r in bertbase deberta modernbert gpt2 gptj shufflen1; do
       experiments/drivers/submit_all.sh $r /scratch/$USER/probes/$r
   done
   ```
5. **Verify:** `sbatch experiments/drivers/09_verify.sh`.

Account `p33044` is capped at 8 concurrent GPU jobs, so the training arrays are
throttled to 6 and a long pending queue is expected rather than a fault.

---

## Environment

```bash
# Always use the explicit interpreter path (conda activate is unreliable on the cluster).
PYTHON=/path/to/conda/envs/sp-env/bin/python

# Verified with transformers 4.57.6 (an older note claimed a 4.38.2 pin; the
# environment used for the from-scratch reproduction runs 4.57.6). Also torch,
# scipy, plotly, simple_slurm, omegaconf, pyyaml.  See requirements.txt.
#
# Tokenizers must be *fast* (character offsets are required by the unified
# alignment). Name the tokenizer explicitly via model.hf_model_name.
```

All non-trivial jobs run on SLURM (CPU: `--partition=short`; GPU:
`--partition=gengpu --gres=gpu:1`).

## Configuring paths

**Start here.** Two files hold every machine-local path; both are git-ignored,
and both ship as templates:

```bash
cp paths.yaml.template paths.yaml     # read by Python
cp paths.sh.template   paths.sh       # sourced by shell scripts
```

They carry the same values in two syntaxes — keep them in sync. Fill in:

| key | what it is |
|---|---|
| `python` / `PYTHON` | absolute path to your interpreter. Use the explicit path; `conda activate` is unreliable under SLURM. |
| `roots.corpus` / `DATA_DIR` | PTB-WSJ CoNLL-X, `ptb3-wsj-{train,dev,test}.conllx` |
| `roots.embeddings`, `roots.alignments` | where extraction and alignment write, one subdirectory per model / tokenizer |
| `roots.models` | where the RoBERTa-Shuffle-n1 fairseq checkpoint was downloaded |
| `roots.out`, `roots.figs` | corpus statistics and figures |
| `roots.results`, `roots.lnconv`, `roots.unified` | the probing runs. Point all three at your own output when reproducing from scratch. |
| `predictors.*` | the two corpus-statistic TSVs, and the fastText `wiki-news-300d-1M.vec` the head entropy is measured in |
| `hf_home` / `HF_HOME` | HuggingFace cache the offline jobs read. Point it at a directory **you own** and populate it with `scripts/predownload_models.py`; a shared cache with 0-byte blobs fails with a JSON decode error that looks nothing like a caching problem. |

Nothing else in the repository contains an absolute path. In particular
[`experiments/paper_runs.yaml`](experiments/paper_runs.yaml) — the manifest of
what each probing run *is*, and what the paper claims for it — deliberately
carries none, so it can be version-controlled and shared while the paths to your
own copies stay local. `scripts/_manifest.py` is the single place that joins the
two, and the drivers and the verification script both go through it.

The SLURM account and partition in the `#SBATCH` directives of
`experiments/drivers/*.sh` cannot be read from the config — edit those directly.

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

- **PTB-WSJ** dependency parses (CoNLL-X). Requires an unmodified PTB3 licence.
  No PTB text is bundled here, in any form: the corpus is LDC-licensed and not
  ours to redistribute. `scripts/convert_conll_to_raw.py` regenerates the
  de-PTBified sentences the extractors consume, from your own copy.
- **Natural-sentence embeddings** for the six runs (HDF5; 13 to 46 checkpoints
  per sentence depending on the model, and 4096-dimensional for GPT-J).
- **RoBERTa-Shuffle-N1** fairseq checkpoint (Sinha et al., EMNLP 2021), which
  `scripts/convert_raw_to_roberta_shufflen1.py` converts to HF layout.
- **fastText `wiki-news-300d-1M.vec`**, the static space the similarity-corrected
  head entropy is computed in.
- **HF model cache** for `bert-base-cased`, `roberta-base`,
  `microsoft/deberta-v3-base`, `answerdotai/ModernBERT-base`, `gpt2` and
  `EleutherAI/gpt-j-6B` — compute nodes are offline, so run
  `scripts/predownload_models.py` first (level-2 step 0). Note
  `transformer_lens` resolves `bert-base-cased` to its canonical id
  `google-bert/bert-base-cased`; the script fetches both.

## Layout

```
structural-probes-labelwise-analysis/
├── structural-probes/     # Bundled upstream Hewitt & Manning library
├── scripts/               # Embedding extraction, alignment, PTB prep, checks
│   ├── regression/               # The WLS regression and its model comparisons
│   ├── figures/                  # Curves, R² heat maps, dendrograms
│   ├── check_embeddings.py       # Embeddings + alignments vs the manifest
│   ├── check_extractor_equivalence.py   # The 1+L and 2+2L paths agree
│   ├── make_probe_config.py      # One probe config, from the manifest
│   └── verify_paper_numbers.py   # Every published number vs the code
├── experiments/
│   ├── paper_runs.yaml           # The six runs, and what the paper claims for each
│   ├── drivers/                  # One SLURM script per pipeline stage
│   ├── bert-base-prd/            # BERT-base probes, ULAS, distance/cluster analysis
│   │   ├── results-hface/        # Saved probe outputs (26 checkpoints)
│   │   ├── figures/              # Generators + rendered paper figures
│   │   ├── tables/               # Regression / range CSV+HTML
│   └── roberta-shufflen1-prd/    # RoBERTa-Shuffle-N1 probes
├── scripts/_manifest.py   # Joins paper_runs.yaml with your local paths.yaml
├── paths.yaml.template    # Copy to paths.yaml (git-ignored) and fill in
├── paths.sh.template      # Copy to paths.sh   (git-ignored) and fill in
├── requirements.txt
├── LICENSE                # Apache 2.0
└── UPSTREAM_README.md     # Original structural-probes README
```

---

## Licence and attribution

Released under the **Apache License 2.0**, the licence of the upstream
[structural-probes](https://github.com/john-hewitt/structural-probes)
repository by John Hewitt, of which `structural-probes/` here is a derivative.

`LICENSE` retains Hewitt's copyright notice, as Apache 2.0 §4 requires, and adds
one for the work in this repository. Modifications to the upstream library are
listed in [`UPSTREAM_README.md`](UPSTREAM_README.md); the substantive one is
`structural-probes/data.py`, which gained a model-agnostic pre-aligned embedding
path that honours each tokenizer's own special-token counts rather than assuming
BERT's `[CLS]`/`[SEP]`.

If you use this code, please cite both the paper and Hewitt & Manning (2019).
