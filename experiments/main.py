from argparse import ArgumentParser

import subprocess

import sys
import os
import time
import logging

from dataclasses import dataclass
from typing import List

from omegaconf import OmegaConf


@dataclass
class ExperimentParams:
    batch_size: int
    seq_len: int
    hidden_size: int
    num_heads: int
    dtype: str
    device: str


@dataclass
class ExperimentConfig:
    experiment: str
    module: str
    params: ExperimentParams


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main(experiments: List[ExperimentConfig]):
    total = len(experiments)

    for i, cfg in enumerate(experiments, 1):
        name = cfg.experiment
        module = cfg.module
        params = cfg.params

        print(f"[{i}/{total}] STARTING: {name}")

        cmd = [sys.executable, "-m", module]

        param_dict = OmegaConf.to_container(params, resolve=True)
        for k, v in param_dict.items():
            cmd.extend([f"--{k}", str(v)])

        cmd.extend(["--experiment_name", name])

        try:
            subprocess.run(cmd, check=True)
            print(f"[{i}/{total}] COMPLETED: {name}")
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{total}] FAILED: {name} (Exit Code: {e.returncode})")

        print("-" * 50)
        time.sleep(1)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="experiments_lizard_attention_pytorch.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Could not find config file: {args.config}")

    with open(args.config, "r", encoding="utf8") as fp:
        content = fp.read()

    raw_docs = content.split("---")
    experiments: List[ExperimentConfig] = []

    schema = OmegaConf.structured(ExperimentConfig)

    for doc in raw_docs:
        clean_doc = doc.strip()
        if clean_doc:
            raw_cfg = OmegaConf.create(clean_doc)
            merged_cfg = OmegaConf.merge(schema, raw_cfg)
            OmegaConf.resolve(merged_cfg)
            experiments.append(merged_cfg)

    main(experiments)
