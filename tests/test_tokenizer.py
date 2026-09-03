from tokenizer.basic_tokenizer import BasicTokenizer


text = "hello world"

tokenizer = BasicTokenizer(text)

encoded = tokenizer.encode("hello world")
decoded = tokenizer.decode(encoded)

print("Vocabulary size:", tokenizer.vocab_size)
print("Encoded:", encoded)
print("Decoded:", decoded)

assert decoded == "hello world"

print("Tokenizer test passed!")