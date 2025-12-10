import yaml
import subprocess
import sys
import os
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

def run_experiments(config_path):
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        print(f"Current working directory: {os.getcwd()}")
        return

    experiments = []
    try:
        with open(config_path, 'r') as f:
            # load_all parses "---" separated blocks
            for doc in yaml.safe_load_all(f):
                if doc:
                    experiments.append(doc)
    except Exception as e:
        print(f"Error reading YAML: {e}")
        return

    total = len(experiments)
    print(f"Found {total} experiments in '{config_path}'. Running sequentially...\n")

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
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
        run_experiments(yaml_file)
    else:
        print("Usage: python -m modules.runner <path_to_yaml>")
        print("Example: python -m modules.runner modules/experiments_gla.yaml")