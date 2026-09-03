import torch
from torch.optim import AdamW

from model.gpt import GPT
from training.data_loader import load_training_data
from training.loss import language_model_loss


# -----------------------------
# Configuration
# -----------------------------

BLOCK_SIZE = 16
BATCH_SIZE = 8
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 3e-4
EPOCHS = 10


# -----------------------------
# Load real dataset
# -----------------------------

loader, tokenizer = load_training_data(
    "data/train.txt",
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE
)

vocab_size = len(tokenizer.vocab)

print("Vocabulary size:", vocab_size)
print("Number of batches:", len(loader))


# -----------------------------
# Create model
# -----------------------------

model = GPT(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=BLOCK_SIZE,
    num_layers=NUM_LAYERS
)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# -----------------------------
# Training loop
# -----------------------------

model.train()

for epoch in range(EPOCHS):

    total_loss = 0.0

    for x, y in loader:

        optimizer.zero_grad()

        logits = model(x)

        loss = language_model_loss(logits, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )


print("Training completed!")