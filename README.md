Lizard Paper Implementation
---

#### Setup

- on ml-institute machines, run `SETUP` once per clone.
    - it setups git user with dedicated token
- on each login, `conda activate jku-ai-practical-project` should be used to activate conda environment.
---

#### Modules

##### GPU Type: GTX 1080Ti (11GB)

```sh
CUDA_VISIBLE_DEVICES=0 python ./modules/matmul_naive_fp16.py
```
##### GPU Type: V100 

```sh
CUDA_VISIBLE_DEVICES=1 time python ./modules/gla_fla_float32_gpu_time_b_16_s_2048_h_32.py > gla_fla_float32_gpu_time_b_16_s_2048_h_32.log 2>&1
```

##### GPU Type: RTX 2080Ti (11GB)

```sh
CUDA_VISIBLE_DEVICES=0 time python ./modules/gla_fla_float16_gpu_time_b_16_s_2048_h_32.py > gla_fla_float16_gpu_time_b_16_s_2048_h_32_rtx_2080.log 2>&1
```

```sh
CUDA_VISIBLE_DEVICES=0 time python ./modules/gla_fla_float32_gpu_time_b_16_s_2048_h_32.py > gla_fla_float32_gpu_time_b_16_s_2048_h_32_rtx_2080.log 2>&1
```
