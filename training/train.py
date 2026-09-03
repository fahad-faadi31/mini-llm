import torch
from torch.optim import AdamW

from model.gpt import GPT
from training.loss import language_model_loss


# Model settings
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

optimizer = AdamW(model.parameters(), lr=3e-4)


# Dummy training data for now
x = torch.randint(0, vocab_size, (8, 16))
y = torch.randint(0, vocab_size, (8, 16))


# Training step
model.train()

optimizer.zero_grad()

logits = model(x)

loss = language_model_loss(logits, y)

loss.backward()

optimizer.step()

print("Loss:", loss.item())
print("Training step completed!")