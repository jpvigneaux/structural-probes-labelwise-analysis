"""Load the run manifest, with machine-local paths merged in from paths.yaml.

Two files, deliberately:

  experiments/paper_runs.yaml   what each probing run IS -- model, checkpoint
                                layout, LayerNorm convention, tokenizer, and the
                                values the paper publishes. Version-controlled,
                                and contains no path.

  paths.yaml                    where those things live on THIS machine.
                                Git-ignored, created from paths.yaml.template.

Keeping them apart is what lets the manifest be committed and shared while the
paths stay local; it also means a reader who clones the repository has exactly
one file to fill in. This module is the single place that joins them, so the
verification script and the drivers cannot disagree about where anything is.
"""

import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / 'experiments' / 'paper_runs.yaml'
PATHS = REPO / 'paths.yaml'


def load(manifest_path=None, paths_path=None):
    """Return the manifest with `roots` and `predictors` filled in from paths.yaml."""
    manifest_path = Path(manifest_path or MANIFEST)
    paths_path = Path(paths_path or PATHS)

    if not manifest_path.exists():
        sys.exit(f'{manifest_path} not found.')
    if not paths_path.exists():
        sys.exit(f'{paths_path} not found. Copy paths.yaml.template to paths.yaml '
                 f'and fill in the paths for your machine.')

    man = yaml.safe_load(open(manifest_path)) or {}
    cfg = yaml.safe_load(open(paths_path)) or {}
    missing = [k for k in ('roots', 'predictors') if k not in cfg]
    if missing:
        sys.exit(f'{paths_path} lacks {missing}; see paths.yaml.template.')

    man['roots'] = cfg['roots']
    man['predictors'] = cfg['predictors']
    return man
