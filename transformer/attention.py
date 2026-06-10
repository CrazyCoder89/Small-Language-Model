import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, head_size, block_size):
        super().__init__()
        self.head_size = head_size

        self.W_q = nn.Linear(d_model, head_size, bias=False)
        self.W_k = nn.Linear(d_model, head_size, bias=False)
        self.W_v = nn.Linear(d_model, head_size, bias=False)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, d_model = x.shape

        Q = self.W_q(x)    # (B, T, head_size)
        K = self.W_k(x)    # (B, T, head_size)
        V = self.W_v(x)    # (B, T, head_size)

        # attention scores
        scores = Q @ K.transpose(-2, -1)    # (B, T, T)
        scores = scores / (self.head_size ** 0.5)

        # causal mask
        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float('-inf')
        )

        # softmax
        weights = F.softmax(scores, dim=-1)    # (B, T, T)

        # weighted sum of values
        output = weights @ V    # (B, T, head_size)

        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            SingleHeadAttention(d_model, self.head_size, block_size)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
if __name__ == "__main__":
    B, T, d_model = 2, 8, 64
    n_heads       = 4
    block_size    = 8

    x = torch.randn(B, T, d_model)

    # test single head
    sha = SingleHeadAttention(d_model, head_size=16, block_size=block_size)
    out = sha(x)
    print(f"SingleHead input:  {x.shape}")
    print(f"SingleHead output: {out.shape}")    # (2, 8, 16)

    # test multi head
    mha = MultiHeadAttention(d_model, n_heads, block_size)
    out = mha(x)
    print(f"MultiHead input:   {x.shape}")
    print(f"MultiHead output:  {out.shape}")    # (2, 8, 64)

    # verify causal mask is working
    print(f"\nAttention mask (first head):")
    print(sha.mask)


    