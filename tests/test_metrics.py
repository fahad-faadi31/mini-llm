import torch

from evaluation.metrics import calculate_perplexity


loss = 2.0
perplexity = calculate_perplexity(loss)

print("Loss:", loss)
print("Perplexity:", perplexity)

assert perplexity > 0
assert abs(perplexity - 7.389) < 0.01

print("Evaluation metrics test passed!")
