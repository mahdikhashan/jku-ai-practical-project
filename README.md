Lizard Kernels 
---

[![Hugging Face](https://img.shields.io/badge/%20HuggingFace-Space-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/nanoman1/lizard-kernel-leaderboard)

#### Setup

- on ml-institute machines, run `SETUP` once per clone.
    - it setups git user with dedicated token
- on each login, `conda activate jku-ai-practical-project` should be used to activate conda environment.

##### Compatible Flash-Attention Package

```sh
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
```

#### Run Experiments

##### GLA bfloat16
```bash
python -m modules.runner experiments/experiments_gla_bfloat16.yaml
```

##### Local Attention
```bash
python -m modules.runner experiments/experiments_local_attention.yaml
```
