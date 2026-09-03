import torch


def create_sequences(tokens, block_size):
    inputs = []
    targets = []

    for i in range(len(tokens) - block_size):
        inputs.append(tokens[i:i + block_size])
        targets.append(tokens[i + 1:i + block_size + 1])

    return torch.tensor(inputs), torch.tensor(targets)