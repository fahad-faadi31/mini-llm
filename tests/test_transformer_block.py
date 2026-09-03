import torch

from model.transformer_block import TransformerBlock


x = torch.randn(2, 8, 64)

block = TransformerBlock(
    embedding_dim=64,
    hidden_dim=256
)

output = block(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == (2, 8, 64)

print("Transformer block test passed!")