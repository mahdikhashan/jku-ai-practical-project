import torch

model = torch.nn.Linear(2048, 2048).to("cpu")

def data_fn():
    return torch.randn(64, 2048).to("cpu")

def forward_fn(model, data):
    return model(data)
