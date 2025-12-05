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

###### GLA

```sh
CUDA_VISIBLE_DEVICES=0 time python ./modules/gla_fla_float16_gpu_time_b_16_s_2048_h_32.py > gla_fla_float16_gpu_time_b_16_s_2048_h_32_rtx_2080.log 2>&1
```

```sh
CUDA_VISIBLE_DEVICES=0 time python ./modules/gla_fla_float32_gpu_time_b_16_s_2048_h_32.py > gla_fla_float32_gpu_time_b_16_s_2048_h_32_rtx_2080.log 2>&1
```

###### Local Attention

```sh
CUDA_VISIBLE_DEVICES=0 time python ./modules/local_attention_float32.py > local_attention_float32_profile_rtx_2080.log 2>&1
```

##### GPU Type: Titan V (12GB)

###### GLA

```sh
CUDA_VISIBLE_DEVICES=4 time python ./modules/gla_fla_float16_gpu_time_b_16_s_2048_h_32.py > gla_fla_float16_gpu_time_b_16_s_2048_h_32_titan_v.log 2>&1
```

###### Local Attention

```sh
CUDA_VISIBLE_DEVICES=4 time python ./modules/local_attention_float32.py > local_attention_float32_profile_titan_v.log 2>&1
```

###### SWA Musings

```sh
CUDA_VISIBLE_DEVICES=4 time python ./modules/swa_musings_benchmark.py > swa_musings_benchmark_profile_titan_v.log 2>&1
```

---

##### Run Commands for All Modules

```sh
python ./modules/gla_fla_bfloat16.py
```

```sh
python ./modules/gla_fla_float16.py
```

```sh
python ./modules/gla_fla_float16_gpu_time.py
```

```sh
python ./modules/gla_fla_float16_gpu_time_b_16_s_2048_h_32.py
```

```sh
python ./modules/gla_fla_float16_torch_profile.py
```

```sh
python ./modules/gla_fla_float16_torch_profile_each_iter.py
```

```sh
python ./modules/gla_fla_float16_torch_profile_top_cuda_kernels.py
```

```sh
python ./modules/gla_fla_float16_torch_profile_top_cuda_kernels_fused_recurrent.py
```

```sh
python ./modules/gla_fla_float32.py
```

```sh
python ./modules/gla_fla_float32_gpu_time_b_16_s_2048_h_32.py
```

```sh
python ./modules/local_attention_float32.py
```

```sh
python ./modules/local_attention_float32_b_16_seq_2048_h_512_w_512.py
```

```sh
python ./modules/matmul_naive_fp16.py
```

```sh
python ./modules/matmul_naive_fp16_fp8.py
```

```sh
python ./modules/matmul_naive_fp16_no_benchmark.py
```

```sh
python ./modules/swa_fzkuji.py
```

```sh
python ./modules/swa_fzkuji_benchmark.py
```

```sh
python ./modules/swa_musings.py
```

```sh
python ./modules/swa_musings_benchmark.py
```

```sh
python ./modules/swa_musings_naive_benchmark.py
```

```sh
python ./modules/swa_musings_strided_benchmark.py
```

```sh
python ./modules/swa_musings_strided_gemini_generated_triton_kernel.py
```

```sh
python ./modules/swa_musings_strided_gemini_generated_triton_kernel_v2.py
```

```sh
python ./modules/swa_musings_strided_gemini_generated_triton_kernel_v2_not_working.py
```

```sh
python ./modules/swa_musings_strided_gemini_generated_triton_kernel_v2_not_working_2.py
```

```sh
python ./modules/swa_musings_strided_gemini_generated_triton_kernel_v2_not_working_3.py
```

```sh
python ./modules/swa_musings_strided_gemini_generated_triton_kernel_v2_working_1.py
```

```sh
python ./modules/swa_musings_strided_gemini_generated_triton_kernel_v2_working_2.py
```

```sh
python ./modules/swa_musings_strided_torch_compiled.py
```

```sh
python ./modules/swa_musings_strided_torch_compiled_max_auto_tune.py
```

```sh
python ./modules/swa_musings_strided_torch_compiled_max_auto_tune_seq_4096.py
```

```sh
python ./modules/swa_musings_strided_torch_compiled_max_auto_tune_seq_8192.py
```

```sh
python ./modules/test_triton_gtx_1080_ti.py
```

```sh
python ./modules/test_triton_pascal_gtx_1080_ti.py
```
