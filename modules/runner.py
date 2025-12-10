import yaml
import subprocess
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def run_experiments(config_path="experiments_gla.yaml"):
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return

    experiments = []
    with open(config_path, 'r') as f:
        # load_all parses "---" separated blocks
        for doc in yaml.safe_load_all(f):
            if doc:
                experiments.append(doc)

    total = len(experiments)
    print(f"Found {total} experiments. Running sequentially...\n")

    for i, exp in enumerate(experiments, 1):
        name = exp.get('experiment', 'Unnamed')
        module = exp.get('module')
        params = exp.get('params', {})

        print(f"[{i}/{total}] STARTING: {name}")

        # Construct Command: python -m module --arg val --experiment_name "Name"
        cmd = [sys.executable, "-m", module]
        
        for k, v in params.items():
            cmd.extend([f"--{k}", str(v)])
        
        cmd.extend(["--experiment_name", name])

        try:
            # subprocess.run is BLOCKING. It waits here until the script finishes.
            subprocess.run(cmd, check=True)
            print(f"[{i}/{total}] COMPLETED: {name}")
            
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{total}] FAILED: {name} (Exit Code: {e.returncode})")
        
        print("-" * 50)
        time.sleep(1) 

if __name__ == "__main__":
    run_experiments()
