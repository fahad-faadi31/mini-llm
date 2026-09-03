import torch

from training.loss import language_model_loss


logits = torch.randn(2, 8, 100)
targets = torch.randint(0, 100, (2, 8))

loss = language_model_loss(logits, targets)

print("Loss:", loss.item())

assert loss.ndim == 0
assert loss.item() > 0

print("Loss test passed!")