import math
import os
import sys
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from char_lm.dataset_bpe import BPEDataset
from transformer import GPTModel


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(ROOT_DIR, "data", "input.txt")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "char_lm", "bpe_1000.json")

BLOCK_SIZE = 128
BATCH_SIZE = 32
BPE_VOCAB_SIZE = 1000

D_MODEL = 256
N_HEADS = 8
N_LAYERS = 8

MAX_STEPS = 10000
EVAL_EVERY = 100
LR = 3e-4
WARMUP_STEPS = 500
MIN_LR = 3e-5
VAL_SPLIT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"


print(f"Device: {DEVICE}")
print(f"AMP enabled: {USE_AMP}")


full_dataset = BPEDataset(
    text_path=DATA_PATH,
    block_size=BLOCK_SIZE,
    vocab_size=BPE_VOCAB_SIZE,
    tokenizer_path=TOKENIZER_PATH,
)

vocab_size = full_dataset.vocab_size

n = len(full_dataset)
n_val = int(n * VAL_SPLIT)
n_train = n - n_val

train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset,
    [n_train, n_val],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=(DEVICE == "cuda"),
    drop_last=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=(DEVICE == "cuda"),
    drop_last=True,
)

print(f"Vocab size:    {vocab_size}")
print(f"Train samples: {n_train:,}")
print(f"Val samples:   {n_val:,}")


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def get_lr(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS

    progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    progress = min(1.0, max(0.0, progress))

    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return MIN_LR + (LR - MIN_LR) * cosine


@torch.no_grad()
def estimate_val_loss(model, val_loader, num_batches=20):
    model.eval()
    losses = []

    for i, (x, y) in enumerate(val_loader):
        if i >= num_batches:
            break

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            _, loss = model(x, y)

        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def generate_from_prompt(
    model,
    dataset,
    prompt,
    max_new_tokens=200,
    temperature=0.8,
):
    model.eval()

    prompt_ids = dataset.encode(prompt)

    if len(prompt_ids) < BLOCK_SIZE:
        pad_len = BLOCK_SIZE - len(prompt_ids)
        input_ids = [0] * pad_len + prompt_ids
    else:
        pad_len = 0
        input_ids = prompt_ids[-BLOCK_SIZE:]

    idx = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -BLOCK_SIZE:]

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, next_id], dim=1)

    model.train()

    generated_ids = idx[0].tolist()[pad_len:]
    return dataset.decode(generated_ids)


def train(model, name):
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

    train_losses = []
    val_losses = []
    lr_history = []
    steps = []

    print(f"\n{'-' * 40}")
    print(f"Model:  {name}")
    print(f"Params: {model.num_parameters():,}")
    print(f"{'-' * 40}")

    gen = infinite_loader(train_loader)
    start = time.time()

    for step in range(1, MAX_STEPS + 1):
        lr = get_lr(step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = next(gen)

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        if step % EVAL_EVERY == 0:
            val_loss = estimate_val_loss(model, val_loader)
            elapsed = time.time() - start

            train_losses.append(loss.item())
            val_losses.append(val_loss)
            lr_history.append(lr)
            steps.append(step)

            print(
                f"  step {step:>5} | "
                f"train {loss.item():.4f} | "
                f"val {val_loss:.4f} | "
                f"lr {lr:.2e} | "
                f"{elapsed:.1f}s"
            )

    print(f"\n-- Samples from {name} --")

    for prompt in ["ROMEO:", "KING RICHARD:", "First Citizen:"]:
        print(f"\nPrompt: '{prompt}'")
        print(generate_from_prompt(model, full_dataset, prompt))

    return steps, train_losses, val_losses, lr_history


model = GPTModel(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
)

steps, train_losses, val_losses, lr_history = train(model, "GPT-8L-256D-BPE1000-AMP-SAN")

torch.save(model.state_dict(), "gpt_8l_256d_bpe1000_amp.pt")
print("\nModel saved to gpt_8l_256d_bpe1000_amp.pt")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(steps, train_losses, label="Train Loss", color="darkorange", linewidth=2)
ax1.plot(steps, val_losses, label="Val Loss", color="steelblue", linewidth=2)
ax1.set_ylabel("Loss")
ax1.set_title("GPT Transformer - BPE1000 Train vs Val Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(steps, lr_history, label="Learning Rate", color="green", linewidth=2)
ax2.set_ylabel("Learning Rate")
ax2.set_xlabel("Step")
ax2.set_title("Learning Rate Schedule")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("transformer_bpe1000_train_val_amp.png", dpi=150)
plt.show()

print("\nPlot saved to transformer_bpe1000_train_val_amp.png")




