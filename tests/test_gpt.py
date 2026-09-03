import torch

from model.gpt import GPT


vocab_size = 100
embedding_dim = 64
hidden_dim = 256
max_seq_len = 32
num_layers = 2

model = GPT(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    max_seq_len=max_seq_len,
    num_layers=num_layers
)

token_ids = torch.randint(
    0,
    vocab_size,
    (2, 16)
)

logits = model(token_ids)

print("Input shape:", token_ids.shape)
print("Logits shape:", logits.shape)

assert logits.shape == (2, 16, vocab_size)

print("GPT model test passed!")