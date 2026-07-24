"""Submit one SLURM GPU job per RoBERTa layer for structural probe training."""

import datetime
from pathlib import Path

import yaml
from simple_slurm import Slurm

# Load user-specific paths from paths.yaml at the repo root.
_PATHS_FILE = Path(__file__).resolve().parents[2] / "paths.yaml"
with open(_PATHS_FILE) as _f:
    _PATHS = yaml.safe_load(_f)

PYTHON      = _PATHS["python"]
REPO_ROOT   = Path(__file__).resolve().parents[2]
EXP_DIR     = REPO_ROOT / "experiments/roberta-shufflen1-prd"
BASE_CONFIG = EXP_DIR / "base_config.yaml"
RESULTS_DIR = EXP_DIR / "results"
PROBE_SCRIPT = REPO_ROOT / "structural-probes/run_experiment.py"
CONFIGS_DIR = EXP_DIR / "configs"
LOG_DIR     = EXP_DIR / "slurm-logs"

CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG) as f:
    base_cfg = yaml.safe_load(f)

# Patch dataset paths from paths.yaml.
base_cfg["dataset"]["corpus"]["root"] = _PATHS["data"]["corpus"]
base_cfg["dataset"]["embeddings"]["root"] = _PATHS["data"]["embeddings"]["roberta_shufflen1"]

# RoBERTa base: layer 0 = embedding layer, layers 1-12 = transformer layers
for layer in range(13):
    layer_pad    = f"{layer:02d}"
    layer_results = RESULTS_DIR / f"layer-{layer_pad}"
    layer_config  = CONFIGS_DIR / f"layer-{layer_pad}.yaml"

    # Write per-layer config
    cfg = base_cfg.copy()
    cfg['model'] = dict(base_cfg['model'])
    cfg['model']['model_layer'] = layer
    cfg['reporting'] = dict(base_cfg['reporting'])
    cfg['reporting']['root'] = str(layer_results)
    layer_results.mkdir(parents=True, exist_ok=True)
    with open(layer_config, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)

    slurm = Slurm(
        account="p33044",
        partition="gengpu",
        gres="gpu:1",
        nodes=1,
        ntasks_per_node=1,
        time=datetime.timedelta(hours=4),
        mem="8G",
        job_name=f"probe-shufflen1-L{layer_pad}",
        output=str(LOG_DIR / "%x-%j.log"),
    )
    slurm.set_shell("/bin/bash")
    slurm.add_cmd(f"export PYTHONPATH={REPO_ROOT}/structural-probes")
    slurm.add_cmd(f"echo '=== Layer {layer} ==='")
    slurm.sbatch(
        f"{PYTHON} {PROBE_SCRIPT} {layer_config}"
        f" --results-dir {layer_results}"
        f" --train-probe 1"
        f" --report-results 1"
    )
    print(f"Submitted layer {layer:2d} → results in {layer_results}")
