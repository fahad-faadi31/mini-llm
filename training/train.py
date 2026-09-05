import torch

from model.gpt import GPT
from training.data_loader import load_training_data
from training.loss import language_model_loss
from evaluation.metrics import calculate_loss, calculate_perplexity


DATA_PATH = "data/train.txt"

BLOCK_SIZE = 16
BATCH_SIZE = 8

EMBEDDING_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2

LEARNING_RATE = 3e-4
EPOCHS = 10

MODEL_PATH = "model.pth"
TOKENIZER_PATH = "tokenizer.json"


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


train_loader, val_loader, tokenizer = load_training_data(
    DATA_PATH,
    block_size=BLOCK_SIZE,
    batch_size=BATCH_SIZE
)

vocab_size = len(tokenizer.vocab)

print("Vocabulary size:", vocab_size)
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


model = GPT(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=BLOCK_SIZE,
    num_layers=NUM_LAYERS
).to(device)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0.0
    train_batches = 0

    for inputs, targets in train_loader:

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)

        loss = language_model_loss(
            logits,
            targets
        )

        loss.backward()

        optimizer.step()

        total_train_loss += loss.item()
        train_batches += 1


    train_loss = total_train_loss / train_batches

    val_loss = calculate_loss(
        model,
        val_loader,
        device=device
    )

    perplexity = calculate_perplexity(
        val_loss
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Train Loss: {train_loss:.4f} - "
        f"Val Loss: {val_loss:.4f} - "
        f"Perplexity: {perplexity:.2f}"
    )


print("Training completed!")


torch.save(
    model.state_dict(),
    MODEL_PATH
)

print(f"Model saved to {MODEL_PATH}")


tokenizer.save(
    TOKENIZER_PATH
)

print(f"Tokenizer saved to {TOKENIZER_PATH}")
