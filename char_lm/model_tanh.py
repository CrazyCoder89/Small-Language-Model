import torch
import torch.nn as nn
import torch.nn.functional as F

class CharMLP(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim=64, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)

        self.net = nn.Sequential(
            nn.Linear(block_size * embedding_dim, hidden_dim),
            nn.Tanh(),                              # ← Tanh
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),                              # ← Tanh
            nn.Linear(hidden_dim, block_size * vocab_size),
        )

    def forward(self, x, targets=None):
        B, T = x.shape
        emb = self.token_embedding(x)
        emb = emb.view(B, -1)
        logits = self.net(emb)
        logits = logits.view(B, T, self.vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, self.vocab_size),
                targets.view(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = CharMLP(vocab_size=65, block_size=8)
    x = torch.randint(0, 65, (4, 8))
    y = torch.randint(0, 65, (4, 8))
    logits, loss = model(x, y)
    print(f"[tanh]  logits: {logits.shape} | loss: {loss.item():.4f} | params: {model.num_parameters():,}")



    