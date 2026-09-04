import torch
from model.gpt import GPT


VOCAB_SIZE = 74
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
MAX_SEQ_LEN = 16
NUM_LAYERS = 2


model = GPT(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    max_seq_len=MAX_SEQ_LEN,
    num_layers=NUM_LAYERS,
)

model.load_state_dict(
    torch.load("model.pth", map_location="cpu")
)

model.eval()

print("Model loaded successfully!")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")