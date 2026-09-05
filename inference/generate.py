import torch

from model.gpt import GPT
from tokenizer.basic_tokenizer import BPETokenizer


MODEL_PATH = "model.pth"
TOKENIZER_PATH = "tokenizer.json"

EMBEDDING_DIM = 64
HIDDEN_DIM = 256
MAX_SEQ_LEN = 16
NUM_LAYERS = 2


def load_model(vocab_size):
    model = GPT(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=NUM_LAYERS,
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu")
    )

    model.eval()

    return model


def generate(model, input_ids, max_new_tokens=20):
    input_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = input_ids[:, -MAX_SEQ_LEN:]

            logits = model(context)

            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

    return input_ids


tokenizer = BPETokenizer.load(TOKENIZER_PATH)

vocab_size = len(tokenizer.vocab)

print("Tokenizer loaded!")
print("Vocabulary size:", vocab_size)
print("Number of merges:", len(tokenizer.merges))

model = load_model(vocab_size)

print("Model loaded!")
print("Generating...")

prompt = "Artificial intelligence"

encoded = tokenizer.encode(prompt)

input_ids = torch.tensor(
    [encoded],
    dtype=torch.long
)

output_ids = generate(
    model,
    input_ids,
    max_new_tokens=20
)

generated_text = tokenizer.decode(
    output_ids[0].tolist()
)

print()
print("Prompt:", prompt)
print("Generated:", generated_text)
