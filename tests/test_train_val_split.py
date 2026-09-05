from training.data_loader import load_training_data


train_loader, val_loader, tokenizer = load_training_data(
    "data/train.txt",
    block_size=16,
    batch_size=8
)

print("Vocabulary size:", len(tokenizer.vocab))
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

train_inputs, train_targets = next(iter(train_loader))
val_inputs, val_targets = next(iter(val_loader))

print("Training input shape:", train_inputs.shape)
print("Validation input shape:", val_inputs.shape)

assert train_inputs.shape[1] == 16
assert val_inputs.shape[1] == 16
assert len(train_loader) > 0
assert len(val_loader) > 0

print("Train/validation split test passed!")
