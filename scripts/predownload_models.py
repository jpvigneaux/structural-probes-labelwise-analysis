"""Pre-download the HuggingFace models needed for embedding extraction.

Compute nodes on the cluster have NO outbound internet, so every model must be
present in the local HF cache before any extraction / probe job runs. Run this
ONCE on a node that has internet (the login node), pointing HF_HOME at the cache
directory the jobs will use:

    HF_HOME=/gpfs/scratch/<user>/hf-cache python scripts/predownload_models.py

Two gotchas this handles, both discovered the hard way:

  1. transformer_lens (used by convert_raw_to_bert_natural_sentences.py) rewrites
     'bert-base-cased' to its CANONICAL repo id 'google-bert/bert-base-cased'
     before calling AutoConfig. A cache that only has the legacy alias misses
     offline and tries the network. We therefore fetch BOTH ids.

  2. A cache dir whose blobs are 0-byte placeholders (e.g. from a failed rsync)
     looks present but fails with a JSON decode error. If a model reports as
     already cached but jobs still fail, delete its models--* dir and re-run.

After this succeeds, set HF_HUB_OFFLINE=1 in the extraction jobs (paths.sh does
this via HF_HOME/HF_HUB_OFFLINE) so they never touch the network.
"""
import os
import sys

from huggingface_hub import snapshot_download

# Legacy alias + canonical id for each model the extraction scripts load.
MODELS = [
    "bert-base-cased",              # AutoTokenizer in convert_raw_to_bert_natural_sentences.py
    "google-bert/bert-base-cased",  # canonical id transformer_lens resolves to
    "roberta-base",                 # RobertaTokenizer/Config in convert_raw_to_roberta_shufflen1.py
    "FacebookAI/roberta-base",      # canonical id
]


def main():
    # This script MUST reach the network; clear offline flags that paths.sh sets
    # for the (offline) compute jobs, in case the user sourced it first.
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)

    hf_home = os.environ.get("HF_HOME", "(unset — using default ~/.cache/huggingface)")
    print(f"HF_HOME = {hf_home}")
    failures = []
    for repo_id in MODELS:
        try:
            path = snapshot_download(repo_id)
            print(f"  OK   {repo_id} -> {path}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAIL {repo_id}: {type(exc).__name__}: {exc}")
            failures.append(repo_id)
    if failures:
        print(f"\n{len(failures)} model(s) failed: {failures}", file=sys.stderr)
        sys.exit(1)
    print("\nAll models cached. Extraction jobs can now run with HF_HUB_OFFLINE=1.")


if __name__ == "__main__":
    main()
