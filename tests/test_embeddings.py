import torch

from model.embeddings import TokenEmbedding


vocab_size = 100
embedding_dim = 64

embedding = TokenEmbedding(vocab_size, embedding_dim)

token_ids = torch.tensor([
    [1, 5, 10, 20],
    [3, 7, 15, 25]
])

output = embedding(token_ids)

print("Input shape:", token_ids.shape)
print("Output shape:", output.shape)

assert output.shape == (2, 4, 64)

print("Embedding test passed!")