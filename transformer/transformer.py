import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, block_size)
        self.ffn       = FeedForward(d_model)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
class GPTModel(nn.Module):
    def __init__(self, vocab_size, block_size, d_model, n_heads, n_layers):
        super().__init__()
        self.block_size = block_size

        self.token_embedding    = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, block_size)
              for _ in range(n_layers)]
        )

        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x, targets=None):
        B, T = x.shape

        tok_emb = self.token_embedding(x)                          # (B, T, d_model)
        pos_emb = self.position_embedding(torch.arange(T, device=x.device))  # (T, d_model)
        x = tok_emb + pos_emb                                      # (B, T, d_model)

        x = self.blocks(x)       # (B, T, d_model)
        x = self.norm_final(x)   # (B, T, d_model)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs  = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
    
if __name__ == "__main__":
    vocab_size  = 65
    block_size  = 32
    d_model     = 64
    n_heads     = 4
    n_layers    = 4

    model = GPTModel(vocab_size, block_size, d_model, n_heads, n_layers)
    print(f"Parameters: {model.num_parameters():,}")

    x = torch.randint(0, vocab_size, (2, block_size))
    y = torch.randint(0, vocab_size, (2, block_size))

    logits, loss = model(x, y)
    print(f"Input shape:   {x.shape}")
    print(f"Logits shape:  {logits.shape}")
    print(f"Loss:          {loss.item():.4f}")


