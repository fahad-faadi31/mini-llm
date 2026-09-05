import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer.basic_tokenizer import BPETokenizer
from training.dataset import create_sequences


class TextDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.inputs, self.targets = create_sequences(tokens, block_size)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def load_training_data(file_path, block_size, batch_size):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer(text)
    tokenizer.train(num_merges=50)

    tokens = tokenizer.encode(text)

    dataset = TextDataset(tokens, block_size)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader, tokenizer
