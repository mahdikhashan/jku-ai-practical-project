import triton
import triton.language as tl
import torch

print(triton.__version__)

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
print("Triton works, output:", out.item())
