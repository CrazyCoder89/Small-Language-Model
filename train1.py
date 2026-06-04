import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

from char_lm.dataset import CharDataset
from char_lm.model_tanh import CharMLP as TanhMLP
from char_lm.model_gelu import CharMLP as GeluMLP

# ── config ────────────────────────────────────────────────
DATA_PATH    = "data/input.txt"
BLOCK_SIZE   = 32
BATCH_SIZE   = 64
EMBEDDING_DIM= 64
HIDDEN_DIM   = 256
MAX_STEPS    = 5000
EVAL_EVERY   = 100
LR           = 3e-4
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")

# ── data ──────────────────────────────────────────────────
dataset = CharDataset(DATA_PATH, BLOCK_SIZE)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=(DEVICE=="cuda"))

def infinite_loader(loader):
    """Loops the dataloader forever — we control steps, not epochs."""
    while True:
        for batch in loader:
            yield batch

# ── training function ─────────────────────────────────────
def train(model, name):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    losses = []         # loss at every eval point
    steps  = []         # step numbers for x-axis

    print(f"\n{'─'*40}")
    print(f"Training: {name}  |  params: {model.num_parameters():,}")
    print(f"{'─'*40}")

    gen = infinite_loader(loader)
    start = time.time()

    for step in range(1, MAX_STEPS + 1):

        x, y = next(gen)
        x, y = x.to(DEVICE), y.to(DEVICE)

        # forward
        logits, loss = model(x, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        if step % EVAL_EVERY == 0:
            elapsed = time.time() - start
            losses.append(loss.item())
            steps.append(step)
            print(f"  step {step:>5} | loss {loss.item():.4f} | {elapsed:.1f}s")

    # generate a sample after training
    print(f"\n── Sample from {name} ──")
    context = torch.zeros((1, BLOCK_SIZE), dtype=torch.long, device=DEVICE)
    generated = model.generate(context, max_new_tokens=200)
    tokens = generated[0].tolist()
    print(dataset.decode(tokens))

    return steps, losses

# ── run both ──────────────────────────────────────────────
tanh_model = TanhMLP(dataset.vocab_size, BLOCK_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
gelu_model = GeluMLP(dataset.vocab_size, BLOCK_SIZE, EMBEDDING_DIM, HIDDEN_DIM)

tanh_steps, tanh_losses = train(tanh_model, "Tanh")
gelu_steps, gelu_losses = train(gelu_model, "GELU")

# ── plot ──────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(tanh_steps, tanh_losses, label="Tanh", color="steelblue",  linewidth=2)
plt.plot(gelu_steps, gelu_losses, label="GELU", color="darkorange", linewidth=2)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Tanh vs GELU — Training Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("tanh_vs_gelu.png", dpi=150)
plt.show()
print("\nPlot saved to tanh_vs_gelu.png")

