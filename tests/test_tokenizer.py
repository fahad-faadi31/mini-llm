from tokenizer.basic_tokenizer import BPETokenizer


text = "hello hello hello"

tokenizer = BPETokenizer(text)
tokenizer.train(num_merges=50)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("Original:", text)
print("Encoded:", encoded)
print("Decoded:", decoded)

assert decoded == text

print("Tokenizer test passed!")