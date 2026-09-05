import torch
from model.gpt import GPT
from tokenizer.basic_tokenizer import BPETokenizer


VOCAB_SIZE = 74
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
MAX_SEQ_LEN = 16
NUM_LAYERS = 2


def load_model():
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
    return model


def generate(model, input_ids, max_new_tokens=20):
    input_ids = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Keep only the latest context window
            context = input_ids[:, -MAX_SEQ_LEN:]

            logits = model(context)

            # Get logits for the last token
            next_token_logits = logits[:, -1, :]

            # Greedy decoding: choose the highest-probability token
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


# Load tokenizer
text = "Artificial intelligence is a field of computer science."
tokenizer = BPETokenizer(text)
tokenizer.train(num_merges=50)

# Load trained model
model = load_model()

# Starting prompt
prompt = "Artificial"

encoded = tokenizer.encode(prompt)

input_ids = torch.tensor(
    [encoded],
    dtype=torch.long
)

# Generate
output_ids = generate(
    model,
    input_ids,
    max_new_tokens=20
)

# Decode
generated_text = tokenizer.decode(
    output_ids[0].tolist()
)

print("Prompt:", prompt)
print("Generated:", generated_text)