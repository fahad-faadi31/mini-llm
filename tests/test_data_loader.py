from training.data_loader import load_training_data


loader, tokenizer = load_training_data(
    "data/train.txt",
    block_size=16,
    batch_size=4
)

x, y = next(iter(loader))

print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Vocabulary size:", len(tokenizer.vocab))

assert x.shape == (4, 16)
assert y.shape == (4, 16)

print("Data loader test passed!")