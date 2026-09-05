import torch
import torch.nn.functional as F


def calculate_loss(model, data_loader, device="cpu"):
    model.eval()

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )

            total_loss += loss.item()
            total_batches += 1

    model.train()

    if total_batches == 0:
        return float("inf")

    return total_loss / total_batches


def calculate_perplexity(loss):
    return torch.exp(torch.tensor(loss)).item()
