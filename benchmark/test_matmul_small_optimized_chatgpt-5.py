# generated with Chat-GPT5
import torch, os

# Ensure MPS backend
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Optional MPS tuning (reduce throttling)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def matmul_pytorch(a, b):
    return torch.matmul(a, b)

class MatmulModel:
    def __init__(self):
        self.a = torch.randn(10000, 10000, device=device, dtype=torch.float16)
        self.b = torch.randn(10000, 10000, device=device, dtype=torch.float16)

    def __call__(self, data):
        return matmul_pytorch(self.a, self.b)

model = MatmulModel()

def data_fn():
    # no new allocations; tensors already on device
    return (None, None)

def forward_fn(model, data):
    torch.mps.synchronize()
    out = model(data)
    torch.mps.synchronize()
    return out

# optional manual warm-up if needed
if __name__ == "__main__":
    for _ in range(3):
        _ = model(None)
    torch.mps.synchronize()
    print("Warm-up complete.")
