import sys

try:
    import triton
    import triton.language as tl
except ImportError:
    print("triton-pascal not installed or cannot be imported.")
    print("Install with:")
    print('pip install --upgrade triton-pascal -i https://sasha0552.github.io/pascal-pkgs-ci/')
    sys.exit(1)

import torch

print("Triton-pascal version:", triton.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print("  ", torch.cuda.get_device_name(i))

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n):
    pid = tl.program_id(0)
    offs = pid
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)

x = torch.randn(1, device="cuda")
y = torch.randn(1, device="cuda")
out = torch.empty_like(x)

add_kernel[(1,)](x, y, out, 1)
print("Triton-pascal works, output:", out.item())
