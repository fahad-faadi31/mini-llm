from tokenizer.basic_tokenizer import BPETokenizer


text = "hello hello hello"

tokenizer = BPETokenizer(text)

tokenizer.train(num_merges=5)

print("Merges:")
for pair, merged in tokenizer.merges.items():
    print(pair, "->", merged)

print("\nVocabulary:")
print(tokenizer.vocab)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print("\nEncoded:", encoded)
print("Decoded:", decoded)

assert decoded == text

print("\nBPE tokenizer test passed!")