import torch

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

def matmul_pytorch(a, b):
    return torch.matmul(a, b)

class MatmulModel:
    def __call__(self, data):
        a, b = data
        return matmul_pytorch(a, b)

model = MatmulModel()

def data_fn():
    a = torch.randn(1024, 1024).to(device)
    b = torch.randn(1024, 1024).to(device)
    return (a, b)

def forward_fn(model, data):
    return model(data)

# Test it
print("Testing...")
for i in range(100):
    data = data_fn()
    result = forward_fn(model, data)
    if i % 10 == 0:
        print(f"Iteration {i} success")

print("Direct test passed!")
