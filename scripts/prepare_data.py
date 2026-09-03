from pathlib import Path
from tokenizer.basic_tokenizer import BPETokenizer


DATA_PATH = Path("data/train.txt")


def load_data():
    return DATA_PATH.read_text(encoding="utf-8")


if __name__ == "__main__":
    text = load_data()

    tokenizer = BPETokenizer(text)
    tokenizer.train(num_merges=50)

    print("Characters:", len(text))
    print("Vocabulary size:", len(tokenizer.vocab))
    print("Number of merges:", len(tokenizer.merges))

    encoded = tokenizer.encode(text)

    print("Total tokens:", len(encoded))
    print("First 50 tokens:", encoded[:50])

    decoded = tokenizer.decode(encoded)

    print("\nDecoded correctly:", decoded == text)