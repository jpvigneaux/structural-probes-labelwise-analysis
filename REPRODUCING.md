# Reproducing the analyses

Every step below runs from this repository. Paths are set once in `paths.sh`.
All non-trivial steps are meant to run on a SLURM compute node, not the login
node.

## 0. The manifest, and how to check you have reproduced anything

The paper reports six probing runs, and they differ from one another in ways that
fail silently if you get them wrong. Every one of those differences is recorded in
[`experiments/paper_runs.yaml`](experiments/paper_runs.yaml), together with the
values the paper publishes for that run, and the drivers in
[`experiments/drivers/`](experiments/drivers/) read it rather than hardcoding
anything.

```bash
# what the paper claims vs. what the code produces, every number, both directions
sbatch experiments/drivers/09_verify.sh
```

That runs two checks. `scripts/check_embeddings.py` validates each run's stored
embeddings and alignment matrices against the manifest — checkpoint count, stored
layout, special-token counts, and the shape agreement that
`align.T @ features[n_pre:-n_suf]` depends on. `scripts/verify_paper_numbers.py`
then refits every regression and diffs it against the published value.

On the archived runs, all six pass the first and **88 of 88 published values
reproduce exactly as printed**.

Five of those values were corrected in the paper after this check was first run.
Each had been read off a script's 4-decimal output and rounded again to 3, which
lands one unit high when the fourth decimal is a 5:

| | was | exact value | now |
| --- | --- | --- | --- |
| BERT-base, R² (3 predictors) | 0.736 | 0.735483 | 0.735 |
| BERT-base, R² with mean(log n) | 0.694 | 0.693470 | 0.693 |
| DeBERTa-v3-base, R² with variance | 0.585 | 0.584492 | 0.584 |
| GPT-2-base, peak dev ULAS | 0.776 | 0.775456 | 0.775 |
| GPT-J-6B, R² with variance | 0.733 | 0.732454 | 0.732 |

The verifier still distinguishes double rounding from a genuine mismatch, and
reports it as its own condition rather than a failure, so the same slip is caught
rather than argued about if it recurs. One knock-on: the appendix reports the
share of explainable variance the predictors capture as a ratio of two printed
values, and `0.735/0.987` reads as 74% at zero decimals while the unrounded ratio
is 75%. That is now given to one decimal (`74.5%`, and `21.8%` for the permuted
control), where the printed and unrounded inputs agree.

---

## 1. Embeddings and alignments

For each model, extract the residual-stream checkpoints on de-PTBified natural
sentences, then precompute the subword-to-PTB alignment matrices. Both stages
have a driver that reads the run's settings from the manifest:

```bash
RUN=gpt2 sbatch experiments/drivers/01_extract.sh
RUN=gpt2 sbatch experiments/drivers/03_align.sh
```

or, to call the scripts directly:

```bash
python scripts/convert_raw_to_hf_all_checkpoints.py \
    $DATA_DIR/ptb3-wsj-train.conllx OUT.hdf5 bert-base-cased \
    --dtype float32 --model-dtype float32

python scripts/precompute_alignments_hf.py \
    $DATA_DIR/ptb3-wsj-train.conllx alignments.train.hdf5 --model-name bert-base-cased
```

`convert_raw_to_hf_all_checkpoints.py` writes `2 + 2L` checkpoints for
architectures with a mid-block state and `1 + L` for parallel-residual ones
(GPT-J), recording the layout as HDF5 attributes.
`precompute_alignments_hf.py` records the number of special tokens the tokenizer
adds. Downstream scripts read those attributes rather than assuming a convention
— BERT-style encoders add `[CLS]`/`[SEP]`, GPT-2 and GPT-J add nothing, and a
hardcoded `features[1:-1]` would silently delete two real tokens from every
GPT sentence.

### Which extractor produced which run

Not every run uses the same one, and the choice renumbers the checkpoints:

| run | extractor | layout | checkpoints | "checkpoint 9" means |
| --- | --- | --- | --- | --- |
| BERT-base | `convert_raw_to_bert_natural_sentences.py` | `2+2L` | 26 | post-block of block 4 |
| ModernBERT-base | `convert_raw_to_hf_all_checkpoints.py` | `2+2L` | 46 | post-block of block 4 |
| GPT-2-base | `convert_raw_to_hf_all_checkpoints.py` | `2+2L` | 26 | post-block of block 4 |
| GPT-J-6B | `convert_raw_to_hf_all_checkpoints.py --allow-missing-midblock` | `1+L` | 29 | post-block of block 9 |
| DeBERTa-v3-base | `convert_raw_to_hf_natural_sentences.py` | `1+L` | 13 | post-block of block 9 |
| RoBERTa-Shuffle-n1 | `convert_raw_to_roberta_shufflen1.py` | `1+L` | 13 | post-block of block 9 |

The two `1+L` encoder runs are not a separate pipeline. For a post-LayerNorm
architecture the `1+L` extraction is exactly the post-block subset of the `2+2L`
one taken under convention B, and that is checked rather than asserted:

```bash
python scripts/check_extractor_equivalence.py --model microsoft/deberta-v3-base \
    --conllx $DATA_DIR/ptb3-wsj-dev.conllx --n-sentences 40
```

It runs all three scripts end to end and compares `B[2k+1]` against
`hidden_states[k+1]`. Over 520 checkpoint comparisons the largest disagreement is
`5.7e-06` for both DeBERTa-v3-base and RoBERTa-base — float32 storage plus one
recomputed LayerNorm. So neither layout is privileged, and
`convert_raw_to_hf_natural_sentences.py` is now restricted to post-LayerNorm
encoders, because for a
pre-LayerNorm model HuggingFace applies the final norm to `hidden_states[-1]` and
to no other entry, which would put the last checkpoint in a different space from
its neighbours.

### LayerNorm convention

`apply_consuming_layernorm.py` converts a stored checkpoint file to the
"what the next sub-layer reads" convention by applying each checkpoint's
consuming LayerNorm. It supports post-LN encoders (BERT, RoBERTa, DeBERTa-v2/v3),
pre-LN decoders (GPT-2), parallel-residual models (GPT-J) and ModernBERT, and it
validates the mapping against the tensors the sub-layers actually receive before
writing anything:

```bash
python scripts/apply_consuming_layernorm.py IN.hdf5 OUT.hdf5 gpt2 \
    --conllx $DATA_DIR/ptb3-wsj-dev.conllx --validate-sentences 25
```

For post-LN encoders the two literature conventions coincide, so this is only
needed to undo the extractor's choice of storing the pre-LayerNorm accumulator.

Which convention each run uses is not a free choice, and it is what makes the
cross-model comparison in the paper like-for-like. In a post-LayerNorm block the
residual stream is normalised in place, so the representation the architecture
maintains *is* the post-LayerNorm value; in a pre-LayerNorm block nothing
normalises the stream itself. Taking "the residual stream as the architecture
maintains it" in both cases therefore means:

| run | convention | why |
| --- | --- | --- |
| BERT-base | B | post-LN; undoes the extractor's pre-LayerNorm accumulator |
| DeBERTa-v3-base | B (already) | post-LN `hidden_states`; no transform needed |
| RoBERTa-Shuffle-n1 | B (already) | post-LN `hidden_states`; no transform needed |
| ModernBERT-base | A | pre-LN; the stream is never normalised in place |
| GPT-2-base | A | pre-LN |
| GPT-J-6B | A | pre-LN, parallel residual |

`experiments/drivers/02_convention.sh` runs this stage for the runs that declare
convention B and exits with nothing to do for the others.

---

## 2. Training probes

`structural-probes/run_experiment.py` with a config using `model_type: hf-disk`
and `alignment: pre-aligned`. One probe per checkpoint:

```bash
RUN=gpt2 RESULTS_ROOT=/scratch/$USER/probes/gpt2 \
    sbatch --array=0-25 experiments/drivers/04_train.sh
```

The config is written by `scripts/make_probe_config.py`, which takes the probe
hyper-parameters (rank 64, L1 loss, 40 epochs, batch 20) from
`experiments/bert-base-prd/base_config_hface_prealigned.yaml` and varies only the
five things that differ between runs: embedding directory, alignment directory,
file stem, hidden dimension and checkpoint index. That identity of
hyper-parameters across models is what the paper's cross-model claim rests on, so
it is worth having one place that enforces it. `hidden_dim` comes from the
manifest rather than being inferred from the model name, because it cannot be:
BERT-base, GPT-2, DeBERTa-v3-base and ModernBERT-base are all 768-dimensional and
GPT-J is 4096. Asking for a checkpoint the run does not have is refused rather
than trained.

---

## 3. Per-relation ULAS

```bash
python scripts/compute_uuas_by_relation.py --results-dir RESULTS --layers 0-25
```

Writes `dev.uuas_by_relation` and `test.uuas_by_relation` (columns:
`relation`, `uuas`, `correct`, `total`) into each `layer-NN/`. CPU-only; training
embeddings are never loaded.

---

## 4. The WLS regression

The published model has three predictors: the similarity-corrected entropy of the
relation's head, the mean of the log arc lengths, and their standard deviation.
Two drivers cover the whole of it — the corpus statistics once, then every
regression the paper reports, per run:

```bash
sbatch experiments/drivers/06_predictors.sh                      # once
RUN=bertbase sbatch experiments/drivers/07_tables.sh             # per run
```

The individual scripts, which is what those drivers call. Three inputs, all on PTB:

```bash
# (a) arc-length moments per relation: mean_length, mean_log_length,
#     stdev_log_length and skew_log_length -- location, dispersion, asymmetry
python scripts/regression/arc_length_moments.py \
    $DATA_DIR/ptb3-wsj-train.conllx --output dep-lengths-ptb.tsv

# (b) similarity-corrected entropy per relation
#     --vec  : external static fastText space (model-independent)
python scripts/regression/contextual_sim_entropy.py \
    $DATA_DIR/ptb3-wsj-train.conllx --out results_sim_ptb.tsv \
    --vec wiki-news-300d-1M.vec --pca-dim 50

#     --embeddings : the probed model's own space, at the probed checkpoint
python scripts/regression/contextual_sim_entropy.py \
    $DATA_DIR/ptb3-wsj-train.conllx --out sim_ctx_ck16.tsv \
    --embeddings EMB.hdf5 --alignments ALIGN.hdf5 --layer 16 --pca-dim 50

# (c) regression -- the three published predictors. Reproduces Table 1 of the
#     paper for BERT-base at checkpoint 16, and one row of the cross-model table
#     for each other run at its own optimal checkpoint. `--no-sd` recovers the
#     two-predictor model of the earlier version of the analysis.
python scripts/regression/ptb_wls_regression.py \
    -uuas RESULTS/layer-16/dev.uuas_by_relation \
    -sim sim_static_fasttext.tsv -len dep-lengths-ptb.tsv --label "BERT-base"
```

The four scripts in `scripts/regression/` each carry their own copy of the join
between these three files, differing only in which columns they build.
`scripts/regression/_regdata.py` is that join written once, and
`scripts/verify_paper_numbers.py` uses it — so if one of those copies ever drifts
from the others, the verification pass is what notices.

### How arc length enters the regression

Three parametrisation choices sit behind `mean(log n)` and `sd(log n)`, each
with a script that reproduces the comparison (all five models, identical data
and weights, so `R^2`, adjusted `R^2` and AIC are directly comparable):

```bash
# mean of the log vs log of the mean          (mean of the log wins, ~0.09 R^2)
python scripts/regression/compare_length_predictors.py -uuas ... -sim ... -len ...

# standard deviation vs variance             (sd wins on all five models)
python scripts/regression/compare_dispersion_scale.py -uuas ... -sim ... -len ...

# does the third moment add anything?        (no: skew insignificant everywhere)
python scripts/regression/add_length_spread.py -uuas ... -sim ... -len ... --full
```

A fourth script bears on how the regression should be *read* rather than on how
it is specified:

```bash
# what share of the between-relation variance in ULAS is real, not sampling
# noise? This caps the attainable R^2, and matters when comparing a model whose
# ULAS is high and well spread with one sitting near a floor.
python scripts/regression/ulas_reliability.py \
    --spec "BERT-base:RESULTS:16" --spec "RoBERTa-Shuffle-n1:RESULTS_SHUF:9"
```

Reliability is 0.98-0.99 for every pre-trained model in the paper and 0.896 for
the permuted-pretraining control, so none of the regressions is noise-limited
and the control's low `R^2` is not a floor effect.

`compare_dispersion_scale.py` matters for how the paper argues. A second-order
Taylor expansion of `E[f(X)]` about the mean introduces dispersion as `sigma^2`,
which would make the variance the indicated regressor. It fits worse in all five
models (`R^2` drops by 0.008-0.025, AIC worsens by 1.1-2.6) and leaves the term
short of significance everywhere (`p` = .056-.179, against .017-.095 for the
standard deviation), so the paper does not rest on that expansion. The sign of
the coefficient is instead argued from mean-preserving spreads, which is exact
and fixes the sign without committing to a parametrisation.

Corpus statistics are computed on **PTB train**: entropy estimates on the
1.7k-sentence dev split would be badly biased, and using train keeps the
predictors independent of the split the ULAS is measured on.

### Why the entropy predictor is computed the way it is

An earlier version of this analysis built the similarity-corrected entropy from
one global sparse similarity matrix. That route is not shipped here, having been
superseded on two counts:

- **Memory.** It materialises one sparse-matrix entry per within-relation word
  pair. On PTB-train that is 756 million pairs (~23 GB) and OOMs; the
  replacement computes cosines in row chunks of a dense per-relation block and
  runs in minutes at 10 GB.
- **Determinism.** Its PCA uses `sklearn`'s `randomized_svd`, whose basis depends
  on a random projection, shifting entropies by ~0.03 bits between runs. The
  replacement uses an exact eigendecomposition of the `d × d` covariance (cheap
  at `d ≈ 300`) with a fixed sign convention.

The entropy mathematics is unchanged: with PCA disabled the two routes agreed
to `1e-6`, i.e. to the printed precision, which is why replacing one with the
other did not move any published number.

---

## 5. Figures

One driver draws every figure for a run, passing that run's own post-block
checkpoint series and reference checkpoint from the manifest:

```bash
RUN=modernbert sbatch experiments/drivers/08_figures.sh
```

The individual scripts:

```bash
# per-relation ULAS across checkpoints
python scripts/plot_selected_relations.py --results-dir RESULTS \
    --out fig.png --model-label "GPT-2-base"

# entropy vs ULAS scatter
python scripts/plot_head_sim_vs_ulas.py --uuas RESULTS/layer-16/dev.uuas_by_relation \
    --sim results_sim_ptb.tsv --out fig.png --model-label "BERT-base (ckpt 16)"

# ULAS-by-distance curves -> NPZ, which feeds the next two
python scripts/figures/uuas_mean_curves_by_checkpoint.py \
    --trained-probes-dir RESULTS --checkpoints 3 5 7 9 11 13 15 17 19 21 23 25 \
    --out curves.html

# two-panel PNG of those curves: all post-block checkpoints on top, one
# highlighted below. Both titles are generated from the NPZ and the arguments,
# so they name the model's own checkpoint set and its own optimum.
python scripts/figures/uuas_mean_curves_png.py --curves curves.npz \
    --out curves.png --model-label "GPT-2-base" \
    --ref-checkpoint 15 --ref-desc "Best post-block checkpoint"

python scripts/figures/regression_uas_vs_log_distance.py --curves curves.npz

python scripts/figures/cluster_relations_by_distance.py --curves curves.npz \
    --alpha 1.0 --out dendrogram.png

# four-panel figure over alpha in {0, 0.33, 0.66, 1}; --range-metric selects the
# corpus component of the composite distance
python scripts/figures/cluster_relations_by_distance.py --curves curves.npz \
    --range-metric w1 --out dendrogram_4panel_w1.png

# diagnostics for that choice: cophenetic correlation, fragmentation, and
# agreement with the pure-ULAS endpoint, per metric and per alpha
python scripts/figures/compare_range_metrics.py --curves curves.npz
```

Pass the post-block checkpoints for the model in question: `3 5 … 2L+1` for the
`2+2L` layout, `1 … L` for the `1+L` layout. `08_figures.sh` reads them from the
manifest's `post_block_checkpoints`, so it cannot mix post-attention and
post-block states into one series:

```bash
$ python experiments/drivers/manifest.py modernbert post_block_checkpoints
3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45
```

Three per-model details that the scripts derive rather than assume, because
getting them wrong is silent:

- **Optimal checkpoint.** `--ref-checkpoint` must name a checkpoint present in
  the NPZ. For a `1+L` model every checkpoint is post-block, so the overall
  optimum is available (DeBERTa-v3-base 9, GPT-J-6B 8). For a `2+2L` model the
  optimum can be a post-attention index that the post-block series excludes
  (GPT-2 16, ModernBERT-base 30); pass the best post-block checkpoint (15, 31)
  and say so with `--ref-desc`.
- **Heat-map width.** `regression_uas_vs_log_distance.py` has one column per
  checkpoint, so it sizes the figure from the checkpoint count. A 28-checkpoint
  model is drawn about twice as wide as a 12-checkpoint one; scaling them to a
  common width in LaTeX instead would compress the printed `R^2` values out of
  legibility.
- **Dendrogram height.** 41 leaves need roughly 7pt of vertical space each. Use
  `--figsize`, `--leaf-font` and `--dpi` to generate at the size the figure will
  be *printed* at, and `--no-title` when several panels share one caption. The
  paper's appendix uses `--figsize 3.02,4.2 --leaf-font 5.0 --dpi 400`.

### Making the figures portable to other pdflatex builds

Matplotlib always writes RGBA PNGs with `tEXt`/`pHYs` chunks. pdfTeX reads an
alpha channel through its soft-mask path, which is where
`pdfTeX error (pdflatex): libpng: internal error` gets reported; whether it
happens depends on the libpng a given TeX distribution was built against, so the
same document can compile on one installation and fail on another. Screenshots
pasted in from macOS are worse: they carry Apple's non-standard `iDOT` chunk.

```bash
# rewrite as opaque 8-bit RGB with only IDAT/IEND, and give files LaTeX-safe
# names; --dry-run first, and check the transparency report before committing
python scripts/figures/flatten_png_for_pdflatex.py --dry-run --rename imgs/*.png
python scripts/figures/flatten_png_for_pdflatex.py --rename \
    --backup-dir /somewhere/imgs-rgba-backup imgs/*.png imgs/appendix/*.png
```

The script refuses to hide a real change: if any file has genuinely transparent
pixels it lists them, because for those, compositing onto white alters the
appearance rather than merely re-encoding. It also flags names containing
parentheses, brackets or spaces, which break some build systems. Every
`\includegraphics` must set an explicit `width`, since stripping `pHYs` discards
the intrinsic-resolution hint.

### Range component of the composite distance

`cluster_relations_by_distance.py --range-metric` chooses how two relations are
compared by the arc lengths they take in the corpus:

- `p90` (default, the published choice): absolute difference of count-weighted
  90th-percentile arc lengths, normalised by the largest.
- `w1`: Wasserstein-1 distance between the arc-length distributions themselves,
  computed exactly as `sum_n |F1(n) - F2(n)|` on the integer grid, in words.

`w1` is the better measurement — it fixes the percentile's instability on
relations that are overwhelmingly local with a thin long tail (`acomp` moves
from the 4th most peripheral relation to rank 19; likewise `predet`, `expl`),
and its dendrograms have cophenetic correlation 0.895-0.899 at every alpha
instead of dipping to 0.842 in the middle. It does *not* make the intermediate
alpha clusters easier to interpret, which was the reason for trying it: the two
metrics correlate at 0.91, both are nearly orthogonal to `d_ULAS` (0.12 and
0.19), and the `w1` partitions at intermediate alpha are more fragmented. The
`alpha = 1` panel, the only one the paper's analysis uses, does not involve the
range component and is identical under both.

---

## 6. The permuted-pretraining control

RoBERTa-Shuffle-n1 (Sinha et al. 2021) is pre-trained on order-permuted
sentences, so it retains distributional but not positional information. Running
the same pipeline on it is the control for whether the log-linear decay and the
regression are properties of an encoded syntax or artefacts of the measurement.
Nothing changes except the pre-trained model, so it is the run key `shufflen1`
and goes through the same drivers as any other:

```bash
experiments/drivers/submit_all.sh shufflen1 /scratch/$USER/probes/shufflen1
```

The weights are a fairseq checkpoint rather than a HuggingFace model, which is
the one thing `01_extract.sh` handles differently for this run:
`convert_raw_to_roberta_shufflen1.py` converts them to HF layout on the fly.
Its tokenizer is `roberta-base`'s, so it shares that alignment directory.

Calling the scripts directly:

```bash
# per-relation ULAS at every checkpoint (CPU)
python scripts/compute_uuas_by_relation.py --results-dir RESULTS_SHUF --layers 0-12

# curves NPZ -> R^2 heat map (RoBERTa is a 1+L layout: checkpoints 1..12)
python scripts/figures/uuas_mean_curves_by_checkpoint.py \
    --trained-probes-dir RESULTS_SHUF --checkpoints 1 2 3 4 5 6 7 8 9 10 11 12 \
    --out curves_shuf.html
python scripts/figures/regression_uas_vs_log_distance.py --curves curves_shuf.npz

# the regression, at its best checkpoint (9)
python scripts/regression/ptb_wls_regression.py \
    -uuas RESULTS_SHUF/layer-09/dev.uuas_by_relation \
    -sim sim_static_fasttext.tsv -len dep-lengths-ptb.tsv --label "RoBERTa-Shuffle-n1"
```

Expected outcome: peak dev ULAS 0.093 (BERT-base 0.815); median `R^2` of the
log-linear decay model 0.102 (0.777); regression `R^2` 0.195 (0.735) with the
arc-length coefficient reduced to -0.002 from -0.252 and both the diversity and
dispersion coefficients reversed in sign. Sampling noise attenuates coefficients
but cannot reverse them, so the sign changes are not a noise artefact — see
`ulas_reliability.py` above.
