from tokenizer.basic_tokenizer import BPETokenizer
from training.dataset import create_sequences


text = "hello world " * 20

tokenizer = BPETokenizer(text)
tokenizer.train(num_merges=10)

tokens = tokenizer.encode(text)

x, y = create_sequences(tokens, block_size=8)

print("Input shape:", x.shape)
print("Target shape:", y.shape)

print("\nInput: ", x[0].tolist())
print("Target:", y[0].tolist())

assert x[0, 1:].tolist() == y[0, :-1].tolist()

print("\nDataset test passed!")