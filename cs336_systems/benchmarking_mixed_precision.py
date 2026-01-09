"""
A simple feed-forward model to benchmark mixed-precision training with autocasting.
"""

import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
fanin, fanout = 5, 2
model = ToyModel(fanin, fanout).to(device)
model.train()
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    input_data = torch.randn(3, fanin).to(device)
    output = model(input_data)
    loss_fn = nn.MSELoss()
    target = torch.randn(3, fanout).to(device)
    loss = loss_fn(output, target)

    # model parameters with the autocast context
    print("Model parameter dtypes under autocast:")
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, dtype: {param.dtype}")

    # output of the first feed-forward layer
    print(f"Output dtype of first layer: {model.fc1(input_data).dtype}")

    # output of layernorm
    print(f"Output dtype of layernorm: {model.ln(model.fc1(input_data)).dtype}")

    # output of the final layer
    print(f"Output dtype of final layer: {output.dtype}")

    # loss
    print(f"Loss dtype: {loss.dtype}")

    # gradients after backward pass
    loss.backward()
    print("Gradient dtypes after backward pass:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"Parameter: {name}, grad dtype: {param.grad.dtype}")
