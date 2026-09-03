import torch.nn.functional as F


def language_model_loss(logits, targets):
    batch_size, seq_len, vocab_size = logits.shape

    logits = logits.view(
        batch_size * seq_len,
        vocab_size
    )

    targets = targets.view(
        batch_size * seq_len
    )

    return F.cross_entropy(logits, targets)