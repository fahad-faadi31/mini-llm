import torch

from model.self_attention import SelfAttention


batch_size = 2
seq_len = 8
embedding_dim = 64

x = torch.randn(batch_size, seq_len, embedding_dim)

attention = SelfAttention(embedding_dim)

output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == (2, 8, 64)

print("Causal self-attention test passed!")