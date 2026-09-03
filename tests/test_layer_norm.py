import torch

from model.layer_norm import LayerNorm


x = torch.randn(2, 8, 64)

layer_norm = LayerNorm(64)

output = layer_norm(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == (2, 8, 64)

print("LayerNorm test passed!")