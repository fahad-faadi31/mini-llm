import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.transformer_block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        max_seq_len,
        num_layers
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size,
            embedding_dim
        )

        self.position_embedding = PositionalEmbedding(
            max_seq_len,
            embedding_dim
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim,
                hidden_dim
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(embedding_dim)

        self.lm_head = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)
        x = self.position_embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits