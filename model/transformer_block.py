import torch.nn as nn

from model.self_attention import SelfAttention
from model.feed_forward import FeedForward
from model.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()

        self.attention = SelfAttention(embedding_dim)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim)

        self.norm1 = LayerNorm(embedding_dim)
        self.norm2 = LayerNorm(embedding_dim)

    def forward(self, x):
        # Attention + residual connection
        x = x + self.attention(self.norm1(x))

        # Feed-forward + residual connection
        x = x + self.feed_forward(self.norm2(x))

        return x