import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer.basic_tokenizer import BPETokenizer


class TextDataset(Dataset):
    def __init__(self, tokens, block_size, stride=None):
        if stride is None:
            stride = block_size

        inputs = []
        targets = []

        for i in range(0, len(tokens) - block_size, stride):
            inputs.append(tokens[i:i + block_size])
            targets.append(tokens[i + 1:i + block_size + 1])

        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def load_training_data(
    file_path,
    block_size,
    batch_size,
    train_ratio=0.9,
    stride=None
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
        block_size,
        stride
    )

    val_dataset = TextDataset(
        val_tokens,
        block_size,
        stride
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, tokenizer
