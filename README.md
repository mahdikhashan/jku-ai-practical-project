Lizard Paper Implementation
---

#### SETUP

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
