Lizard Kernels
---

#### Setup

- on ml-institute machines, run `SETUP` once per clone.
    - it setups git user with dedicated token
- on each login, `conda activate jku-ai-practical-project` should be used to activate conda environment.

##### Compatible Flash-Attention Package

```sh
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post2/flash_attn-2.7.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
```

### Early Results

<details>
  <summary><strong>Benchmark: Lizard, AWA and GLA(FLA)</strong></summary>

![bench](./docs/benchmark_results.png)
![bench-2](./docs/comparison_optimized_final.png)

</details>
