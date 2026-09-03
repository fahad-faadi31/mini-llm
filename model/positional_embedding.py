import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embedding_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embedding_dim
        )

    def forward(self, token_embeddings):
        seq_len = token_embeddings.size(1)

        positions = torch.arange(
            seq_len,
            device=token_embeddings.device
        )

        return token_embeddings + self.position_embedding(positions)