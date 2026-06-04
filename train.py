import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

from char_lm.dataset import CharDataset
from char_lm.model_gelu import CharMLP  # GELU only from now on

# ── config ────────────────────────────────────────────────
DATA_PATH     = "data/input.txt"
BLOCK_SIZE    = 32
BATCH_SIZE    = 64
EMBEDDING_DIM = 64
HIDDEN_DIM    = 256
MAX_STEPS     = 5000
EVAL_EVERY    = 100
LR            = 3e-4
VAL_SPLIT     = 0.1      # NEW: 10% of data for validation
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
# ──────────────────────────────────────────────────────────

print(f"Device: {DEVICE}")
# ── data split ────────────────────────────────────────────
full_dataset = CharDataset(DATA_PATH, BLOCK_SIZE)
vocab_size   = full_dataset.vocab_size

# split indices
n        = len(full_dataset)
n_val    = int(n * VAL_SPLIT)
n_train  = n - n_val

# NEW: split into train and val
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [n_train, n_val]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE == "cuda"),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,                   # no need to shuffle validation
    pin_memory=(DEVICE == "cuda"),
)

print(f"Train samples: {n_train:,}  |  Val samples: {n_val:,}")
# ── infinite loader ───────────────────────────────────────
def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

# ── NEW: validation loss ──────────────────────────────────
@torch.no_grad()
def estimate_val_loss(model, val_loader, num_batches=20):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= num_batches:
            break
        x, y   = x.to(DEVICE), y.to(DEVICE)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

@torch.no_grad()
def generate_from_prompt(model, dataset, prompt, max_new_tokens=300, temperature=0.8):
    model.eval()

    ids = dataset.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        if idx_cond.shape[1] < BLOCK_SIZE:
            pad_len = BLOCK_SIZE - idx_cond.shape[1]
            pad = torch.zeros((1, pad_len), dtype=torch.long, device=DEVICE)
            idx_cond = torch.cat([pad, idx_cond], dim=1)

        logits, _ = model(idx_cond)

        logits = logits[:, -1, :]
        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    model.train()

    return dataset.decode(idx[0].tolist())

# ── training function ─────────────────────────────────────
def train(model, name):
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    train_losses = []
    val_losses   = []
    steps        = []

    print(f"\n{'-'*40}")
    print(f"Training: {name}  |  params: {model.num_parameters():,}")
    print(f"{'-'*40}")

    gen   = infinite_loader(train_loader)
    start = time.time()

    for step in range(1, MAX_STEPS + 1):

        x, y = next(gen)
        x, y = x.to(DEVICE), y.to(DEVICE)

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % EVAL_EVERY == 0:
            val_loss = estimate_val_loss(model, val_loader)   # NEW
            elapsed  = time.time() - start

            train_losses.append(loss.item())
            val_losses.append(val_loss)                        # NEW
            steps.append(step)

            print(f"  step {step:>5} | train {loss.item():.4f} | val {val_loss:.4f} | {elapsed:.1f}s")

    # generate from multiple prompts
    print(f"\n-- Samples from {name} --")
    for prompt in ["ROMEO:", "KING RICHARD:", "First Citizen:"]:
        print(f"\nPrompt: '{prompt}'")
        print(generate_from_prompt(model, full_dataset, prompt))

    return steps, train_losses, val_losses

# ── run ───────────────────────────────────────────────────
model = CharMLP(vocab_size, BLOCK_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
steps, train_losses, val_losses = train(model, "GELU-MLP")

# ── plot ──────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(steps, train_losses, label="Train Loss", color="darkorange", linewidth=2)
plt.plot(steps, val_losses,   label="Val Loss",   color="steelblue",  linewidth=2)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("GELU MLP — Train vs Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mlp_train_val.png", dpi=150)
plt.show()
print("\nPlot saved to mlp_train_val.png")





