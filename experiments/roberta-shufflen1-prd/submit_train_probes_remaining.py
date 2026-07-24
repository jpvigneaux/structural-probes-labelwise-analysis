"""Submit SLURM GPU jobs for layers 1-12, with at most 6 concurrent GPU nodes.

Strategy: two waves of 6 jobs each.
  Wave 1 (layers 1-6):  submitted immediately, run in parallel.
  Wave 2 (layers 7-12): each job depends on afterok:<wave1_peer>, so wave 2
                         job i starts only after wave 1 job i finishes.
                         This keeps the GPU count at most 6 at any time.
"""

import datetime
from pathlib import Path

import yaml
from simple_slurm import Slurm

# Load user-specific paths from paths.yaml at the repo root.
_PATHS_FILE = Path(__file__).resolve().parents[2] / "paths.yaml"
with open(_PATHS_FILE) as _f:
    _PATHS = yaml.safe_load(_f)

PYTHON       = _PATHS["python"]
REPO_ROOT    = Path(__file__).resolve().parents[2]
EXP_DIR      = REPO_ROOT / "experiments/roberta-shufflen1-prd"
BASE_CONFIG  = EXP_DIR / "base_config.yaml"
RESULTS_DIR  = EXP_DIR / "results"
PROBE_SCRIPT = REPO_ROOT / "structural-probes/run_experiment.py"
CONFIGS_DIR  = EXP_DIR / "configs"
LOG_DIR      = EXP_DIR / "slurm-logs"

CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_CONFIG) as f:
    base_cfg = yaml.safe_load(f)

# Patch dataset paths from paths.yaml.
base_cfg["dataset"]["corpus"]["root"] = _PATHS["data"]["corpus"]
base_cfg["dataset"]["embeddings"]["root"] = _PATHS["data"]["embeddings"]["roberta_shufflen1"]

def submit_layer(layer: int, dependency_job_id: int = None) -> int:
    """Submit one probe-training job; return the SLURM job ID."""
    layer_pad     = f"{layer:02d}"
    layer_results = RESULTS_DIR / f"layer-{layer_pad}"
    layer_config  = CONFIGS_DIR / f"layer-{layer_pad}.yaml"

    # Write per-layer config (overwrite if already exists)
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
    if dependency_job_id is not None:
        slurm.set_dependency(f"afterok:{dependency_job_id}")
    slurm.set_shell("/bin/bash")
    slurm.add_cmd(f"export PYTHONPATH={REPO_ROOT}/structural-probes")
    slurm.add_cmd(f"echo '=== Layer {layer} ==='")
    job_id = slurm.sbatch(
        f"{PYTHON} {PROBE_SCRIPT} {layer_config}"
        f" --results-dir {layer_results}"
        f" --train-probe 1"
        f" --report-results 1",
        convert=True,  # return job ID as int
    )
    return job_id

# Wave 1: layers 1-6, submitted immediately (no dependency)
wave1_ids = {}
for layer in range(1, 7):
    job_id = submit_layer(layer)
    wave1_ids[layer] = job_id
    print(f"Wave 1 — layer {layer:2d} submitted → job {job_id}  (results: {RESULTS_DIR / f'layer-{layer:02d}'})")

# Wave 2: layers 7-12, each depends on the wave-1 peer finishing
# layer 7 waits for layer 1, layer 8 waits for layer 2, ..., layer 12 waits for layer 6
for layer in range(7, 13):
    peer = layer - 6          # the wave-1 job this one waits for
    peer_job_id = wave1_ids[peer]
    job_id = submit_layer(layer, dependency_job_id=peer_job_id)
    print(f"Wave 2 — layer {layer:2d} submitted → job {job_id}  (after job {peer_job_id} / layer {peer})")
