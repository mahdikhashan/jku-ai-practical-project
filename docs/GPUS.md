| Rank | GPU               | VRAM         | Arch        | Tensor Cores | Triton | CUDA Cap | #SM | FP16 TFLOPS | FP32 TFLOPS | BF16? | Mem BW   | L2 Cache | L1 per SM |
| ---- | ----------------- | ------------ | ----------- | ------------ | ------ | -------- | --- | ----------- | ----------- | ----- | -------- | -------- | --------- |
| 1    | L4                | 24 GB GDDR6  | Ada Lovelace| Yes          | Yes    | 8.9      | 58  | 121         | 30.3        | Yes   | 300 GB/s | 48 MB    | 128 KB    |
| 2    | GV100 / V100 32GB | 32 GB HBM2   | Volta       | Yes          | Yes    | 7.0      | 80  | 112         | 15.7        | No    | 900 GB/s | 6 MB     | 128 KB    |
| 3    | Tesla V100 16GB   | 16 GB HBM2   | Volta       | Yes          | Yes    | 7.0      | 80  | 112         | 14          | No    | 900 GB/s | 6 MB     | 128 KB    |
| 4    | Titan V           | 12 GB HBM2   | Volta       | Yes          | Yes    | 7.0      | 80  | 110         | 13.8        | No    | 652 GB/s | 4.5 MB   | 128 KB    |
| 5    | RTX 2080 Ti       | 11 GB GDDR6  | Turing      | Yes          | Yes    | 7.5      | 68  | 26.9        | 13.4        | No    | 616 GB/s | 5.5 MB   | 64 KB     |
| 6    | Tesla P100        | 16 GB HBM2   | Pascal      | No           | No     | 6.0      | 56  | 9.3         | 9.3         | No    | 732 GB/s | 4 MB     | 64 KB     |
| 7    | Tesla P40         | 24 GB GDDR5  | Pascal      | No           | No     | 6.1      | 30  | 12          | 12          | No    | 346 GB/s | 2 MB     | 64 KB     |
| 8    | Titan X (Pascal)  | 12 GB GDDR5X | Pascal      | No           | No     | 6.1      | 28  | 11          | 11          | No    | 480 GB/s | 3 MB     | 64 KB     |
| 9    | GTX 1080 Ti       | 11 GB GDDR5X | Pascal      | No           | No     | 6.1      | 28  | 11.3        | 11.3        | No    | 484 GB/s | 2.8 MB   | 64 KB     |
| 10   | GTX 1080 Ti       | 11 GB GDDR5X | Pascal      | No           | No     | 6.1      | 28  | 11.3        | 11.3        | No    | 484 GB/s | 2.8 MB   | 64 KB     |
