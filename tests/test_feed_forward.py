import torch

from model.feed_forward import FeedForward


x = torch.randn(2, 8, 64)

feed_forward = FeedForward(
    embedding_dim=64,
    hidden_dim=256
)

output = feed_forward(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == (2, 8, 64)

print("Feed-forward test passed!")