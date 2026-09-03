import torch

from model.positional_embedding import PositionalEmbedding


batch_size = 2
seq_len = 8
embedding_dim = 64

x = torch.randn(batch_size, seq_len, embedding_dim)

pos_embedding = PositionalEmbedding(
    max_seq_len=128,
    embedding_dim=embedding_dim
)

output = pos_embedding(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == (2, 8, 64)

print("Positional embedding test passed!")