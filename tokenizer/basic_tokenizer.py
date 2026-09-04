import json
from collections import Counter


class BPETokenizer:
    def __init__(self, text):
        self.text = text
        self.vocab = {}
        self.merges = {}

    def get_pairs(self, tokens):
        pairs = Counter()

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] += 1

        return pairs

    def train(self, num_merges=10):
        tokens = list(self.text)

        for _ in range(num_merges):
            pairs = self.get_pairs(tokens)

            if not pairs:
                break

            best_pair, count = pairs.most_common(1)[0]

            new_token = "".join(best_pair)

            self.merges[best_pair] = new_token

            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        unique_tokens = sorted(set(tokens))

        self.vocab = {
            token: i
            for i, token in enumerate(unique_tokens)
        }

    def encode(self, text):
        tokens = list(text)

        for pair, merged_token in self.merges.items():
            new_tokens = []
            i = 0

            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == pair[0]
                    and tokens[i + 1] == pair[1]
                ):
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return [self.vocab[token] for token in tokens]

    def decode(self, token_ids):
        id_to_token = {
            i: token
            for token, i in self.vocab.items()
        }

        return "".join(id_to_token[i] for i in token_ids)

    def save(self, file_path):
        data = {
            "vocab": self.vocab,
            "merges": [
                {
                    "pair": list(pair),
                    "token": merged_token
                }
                for pair, merged_token in self.merges.items()
            ]
        }

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path):
        tokenizer = cls("")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer.vocab = {
            token: int(token_id)
            for token, token_id in data["vocab"].items()
        }

        tokenizer.merges = {
            tuple(item["pair"]): item["token"]
            for item in data["merges"]
        }

        return tokenizer
