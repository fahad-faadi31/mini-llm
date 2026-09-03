from tokenizer.basic_tokenizer import BasicTokenizer


text = "hello world"

tokenizer = BasicTokenizer(text)

print("Vocabulary:")
for token, token_id in tokenizer.stoi.items():
    print(repr(token), "->", token_id)

print("\nVocabulary size:", tokenizer.vocab_size)