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


def load_training_data(
    file_path,
    block_size,
    batch_size,
    train_ratio=0.9
):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer(text)
    tokenizer.train(num_merges=50)

    tokens = tokenizer.encode(text)

    split_index = int(len(tokens) * train_ratio)

    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]

    train_dataset = TextDataset(
        train_tokens,
        block_size
    )

    val_dataset = TextDataset(
        val_tokens,
        block_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, tokenizer
