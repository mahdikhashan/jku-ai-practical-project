import torch

def matmul_pytorch(a, b):
    return torch.matmul(a, b)

class MatmulModel:
    def __call__(self, data):
        a, b = data
        return matmul_pytorch(a, b)

model = MatmulModel()

def data_fn():
    # Start with smaller matrices (256x256)
    a = torch.randn(10000, 10000).to(device)
    b = torch.randn(10000, 10000).to(device)
    return (a, b)

def forward_fn(model, data):
    return model(data)
