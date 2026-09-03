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
        # Start with individual characters
        tokens = list(self.text)

        for _ in range(num_merges):
            pairs = self.get_pairs(tokens)

            if not pairs:
                break

            # Most frequent pair
            best_pair, count = pairs.most_common(1)[0]

            # Create a new token
            new_token = "".join(best_pair)

            self.merges[best_pair] = new_token

            # Merge the pair
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

        # Build vocabulary
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