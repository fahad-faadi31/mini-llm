import torch
from model.gpt import GPT
from training.data_loader import load_training_data
from evaluation.metrics import calculate_loss, calculate_perplexity


DATA_PATH = "data/train.txt"
BLOCK_SIZE = 64
BATCH_SIZE = 64
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 3e-4
EPOCHS = 10
MODEL_PATH = "model.pth"
TOKENIZER_PATH = "tokenizer.json"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


train_loader, val_loader, tokenizer = load_training_data(
    DATA_PATH,
    BLOCK_SIZE,
    BATCH_SIZE,
    stride=BLOCK_SIZE
)

vocab_size = len(tokenizer.vocab)

print(f"Vocabulary size: {vocab_size}")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")


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


best_val_loss = float("inf")


for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    val_loss = calculate_loss(
        model,
        val_loader,
        device=device
    )

    perplexity = calculate_perplexity(val_loss)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f} - "
        f"Val Loss: {val_loss:.4f} - "
        f"Perplexity: {perplexity:.2f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"  ? Best model saved! "
            f"Val Loss: {best_val_loss:.4f}"
        )


tokenizer.save(TOKENIZER_PATH)

print("Training completed!")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Tokenizer saved to {TOKENIZER_PATH}")
