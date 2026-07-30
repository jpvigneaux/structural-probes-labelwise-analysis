# Drivers

One SLURM script per pipeline stage. Every stage takes the run key in `$RUN` and
reads that run's model, layout, paths and checkpoint list from
[`../paper_runs.yaml`](../paper_runs.yaml) — so a driver and the verification
script cannot disagree about which run they are talking about.

```bash
$ python experiments/drivers/manifest.py --keys
bertbase deberta modernbert gpt2 gptj shufflen1
```

## Order

```bash
# once, not per model -- the corpus statistics are shared by every row of the
# cross-model table
sbatch experiments/drivers/06_predictors.sh

# per model: submits 01, 02, 03, 04, 05, 07, 08 with dependencies wired up
experiments/drivers/submit_all.sh gpt2 /scratch/$USER/probes/gpt2

# when every run has finished
sbatch experiments/drivers/09_verify.sh
```

| stage | what it does | resource |
| --- | --- | --- |
| `01_extract.sh` | residual-stream checkpoints, all three splits | GPU |
| `02_convention.sh` | apply each checkpoint's consuming LayerNorm (convention B only) | CPU |
| `03_align.sh` | subword→PTB alignment matrices | CPU |
| `04_train.sh` | one structural probe per checkpoint (array) | GPU |
| `05_ulas.sh` | per-relation ULAS, dev and test (array) | CPU |
| `06_predictors.sh` | arc-length moments and head entropy on PTB train | CPU |
| `07_tables.sh` | every regression the paper reports | CPU |
| `08_figures.sh` | curves, R² heat map, dendrogram | CPU |
| `09_verify.sh` | diff all of it against the published values | CPU |

## Three things the manifest exists to prevent

**Checkpoint renumbering.** Four runs use the `2+2L` layout, which interleaves a
post-attention checkpoint between consecutive post-block ones; DeBERTa-v3-base and
RoBERTa-Shuffle-n1 use `1+L`. "Checkpoint 9" therefore means the ninth block for
those two and the post-attention state of block 4 for the others. Nothing raises
if you get this wrong — the probe trains, and reports a plausible UUAS.
`scripts/check_extractor_equivalence.py` shows the two paths agree where they
overlap (max difference 5.7e-06 for both DeBERTa-v3-base and RoBERTa-base), so
the `1+L` runs are the post-block subset of the `2+2L` ones under convention B
and neither layout is privileged.

**LayerNorm convention.** For a post-LayerNorm encoder the residual stream is
normalised in place, so the stored representation is the post-LayerNorm value;
for a pre-LayerNorm model it never is. Taking "the residual stream as the
architecture maintains it" therefore means convention B for BERT-base and
DeBERTa-v3-base and convention A for GPT-2, ModernBERT and GPT-J. That is what
makes the cross-model comparison like-for-like, and it is why `02_convention.sh`
runs for some models and not others.

**Special tokens.** BERT-family tokenizers add `[CLS]`/`[SEP]`; GPT-2 and GPT-J
add nothing. The alignment file records the counts and `data.BERTDataset` honours
them, so a hardcoded `features[1:-1]` cannot silently delete two real tokens.
`scripts/check_embeddings.py` checks this, along with the shape agreement that
`align.T @ features[n_pre:-n_suf]` depends on.
