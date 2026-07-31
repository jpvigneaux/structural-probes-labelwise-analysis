import datetime
from simple_slurm import Slurm
from pathlib import Path
import yaml

with open(Path(__file__).parents[2] / "paths.yaml", 'r') as file:
   paths = yaml.safe_load(file)

REPO_ROOT = paths['repo_root']
EXPERIMENT_DIR=f"{REPO_ROOT}/experiments/bert-base-prd"
BASE_CONFIG=f"{EXPERIMENT_DIR}/base_config_hface_prealigned.yaml"
RESULTS_DIR=f"{EXPERIMENT_DIR}/results-hface"
PROBE_SCRIPT=f"{REPO_ROOT}/structural-probes/run_experiment.py"

PYTHONPATH=paths['python']


for layer in range(16,26): 
    LAYER_PAD='{:02d}'.format(layer)
    LAYER_RESULTS=f"{RESULTS_DIR}/layer-{LAYER_PAD}"
    Path(LAYER_RESULTS).mkdir(parents=True, exist_ok=True)
    LAYER_CONFIG=f"{LAYER_RESULTS}/probe_bertbase_prealigned_layer{LAYER_PAD}.yaml"

    with open(BASE_CONFIG, 'r') as f:
        cfg = yaml.safe_load(f)
        cfg['model']['model_layer'] = layer
        cfg['dataset']['corpus']['root'] = paths['data']['corpus']
        cfg['dataset']['embeddings']['root'] = paths['data']['embeddings']['bert_base_natural']
        cfg['model']['alignment_root'] = paths['data']['alignment']['bert_base_natural']
        with open(LAYER_CONFIG, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    slurm = Slurm(
        account="p33044",
        partition="gengpu",
        gres="gpu:1",
        nodes=1,
        ntasks_per_node=1,
        time=datetime.timedelta(hours=1),
        mem="8G",
        job_name=f"bert-nat-layer{layer}",
        output=fr'''{REPO_ROOT}/experiments/bert-base-prd/slurm-logs/%x-%j.log'''
    )
    slurm.set_shell("/bin/bash")
    slurm.add_cmd(r"echo $(date '+%Y-%m-%d %H:%M:%S')")
    slurm.add_cmd(f"Begining train of layer-{layer} probe...")
    slurm.sbatch(f"{PYTHONPATH} {PROBE_SCRIPT} {LAYER_CONFIG} --results-dir {LAYER_RESULTS} --train-probe 1 --report-results 1")
 