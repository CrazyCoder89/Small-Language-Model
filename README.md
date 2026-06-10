# SLM — Small Language Model from Scratch

Built a character-level language model from scratch in PyTorch, progressing from a simple MLP baseline to a GPT-style transformer. Every component implemented manually — no HuggingFace, no `nn.Transformer`, no shortcuts.

**Dataset:** Tiny Shakespeare (1.1M characters)  
**Hardware:** NVIDIA RTX 3050 6GB Laptop GPU  
**Framework:** PyTorch 2.7, CUDA 12.8

---

## Project Structure

```
slm/
├── data/
│   └── input.txt                  ← Tiny Shakespeare corpus
├── char_lm/                       ← Phase 0: MLP baseline
│   ├── dataset.py                 ← Tokenizer + Dataset + DataLoader
│   ├── model_tanh.py              ← MLP with Tanh activation
│   ├── model_gelu.py              ← MLP with GELU activation
│   └── __init__.py
├── transformer/                   ← Phase 1: GPT-style transformer
│   ├── attention.py               ← Single + Multi-head attention
│   ├── transformer.py             ← TransformerBlock + GPTModel
│   ├── train.py                   ← GPT-6L-128D training script
│   ├── train1.py                  ← GPT-8L-256D-AMP training script
│   └── __init__.py
└── train.py                       ← MLP training + Tanh vs GELU experiment
```

---

## Phase 0 — MLP Baseline

### What Was Built

A character-level MLP language model trained on Shakespeare. Built to understand the training loop, data pipeline, and the fundamental limitations of flat context modeling before building the transformer.

**Architecture:**
```
Token IDs → Embedding → Flatten → Linear → Activation → Linear → Logits
```

**Key insight:** Flattening destroys all positional relationships between tokens. The model sees 32 characters mashed into one vector — it cannot know that token at position 0 relates to token at position 31. This is exactly the limitation attention solves.

### Experiment: Tanh vs GELU

Ran a controlled ablation — identical architecture, only activation function changed.

| Activation | Step 100 Loss | Final Loss | Notes |
|---|---|---|---|
| Tanh | 2.57 | 0.27 | Collapsed to `&&&&` at larger size |
| GELU | 1.93 | 0.21 | Stable, consistent convergence |

**Finding:** At small scale both tied. At larger scale (block_size=32, 1.1M params) Tanh neurons saturated and collapsed — generating repeating garbage characters. GELU maintained stable gradients throughout. This empirically confirmed why every modern transformer uses GELU.

**GELU:** Gaussian Error Linear Unit. Soft probabilistic gate — scales input by the probability it's positive. Unbounded positive side means gradients stay healthy for large activations, unlike Tanh which squashes everything to (-1, 1).

### MLP Final Results

```
Params:        1,129,056
Block size:    32
Val loss:      0.21
Training time: 19 seconds (GPU)
Generation:    correct structure, gibberish words
```

---

## Phase 1 — GPT-Style Transformer

### Architecture

Built every component from scratch:

**Single-Head Attention:**
```
Q = x @ W_q        (what am I looking for?)
K = x @ W_k        (what do I contain?)
V = x @ W_v        (what do I give if selected?)

scores  = Q @ K.T / sqrt(head_size)
scores  = masked_fill(future positions → -inf)
weights = softmax(scores)
output  = weights @ V
```

**Causal Mask:** Lower triangular matrix. Token at position `t` can only attend to positions `0..t`. Future tokens are set to `-inf` before softmax → become exactly `0` after softmax. Prevents the model from "cheating" by looking ahead during training.

**Multi-Head Attention:** Runs `n_heads` attention operations in parallel, each learning different relationship types (syntax, semantics, position). Outputs concatenated and projected back to `d_model`.

**Transformer Block:**
```python
x = x + attention(layernorm(x))   # pre-norm residual
x = x + feedforward(layernorm(x)) # pre-norm residual
```

**Residual connections:** Each block adds a small correction to `x` rather than replacing it. Original information flows through every layer untouched — enables training 8+ layers deep without vanishing gradients.

**LayerNorm:** Normalizes each token vector to mean=0, std=1 before each sub-layer. Keeps activations stable through many layers. Pre-norm (normalize before attention) is more stable than post-norm.

**Feedforward Network:** Two linear layers with 4x expansion. `d_model → 4*d_model → d_model`. Each token processed independently — this is where the model "thinks" after attention decides which tokens to look at.

**Positional Embedding:** Learned position vectors added to token embeddings. Each of the `block_size` positions has its own `d_model`-dimensional vector. Gives the model explicit knowledge of token order — unlike the MLP which destroyed position information.

### Training Infrastructure

**AdamW:** Adaptive per-weight learning rates (Adam) + correct weight decay (W). Each weight gets its own effective step size based on gradient history. Used by every modern LLM.

**Gradient Clipping:** After `backward()`, before `step()`. If total gradient magnitude exceeds 1.0, scale all gradients down proportionally. Prevents any single bad batch from destroying training.

**Warmup + Cosine LR Schedule:**
```
Steps 1 → 500:     linear warmup  (0 → max_lr)
Steps 500 → 10000: cosine decay   (max_lr → min_lr)
```
Warmup prevents early instability when gradients are unreliable. Cosine decay lets the model settle precisely into a minimum at the end.

**Mixed Precision (AMP):** Forward pass in float16, backward in float32, weights in float32. Cuts activation memory ~40%, faster on tensor cores. Gradient scaling prevents float16 underflow — tiny gradients multiplied by large scale factor, then divided back after backward pass.

**Validation Loss:** 10% of data held out — never trained on. Measured every 100 steps by averaging loss over 20 random validation batches. Confirms generalization vs memorization.

---

## Experiment Results — Full Progression

| Model | Params | Block Size | Steps | Val Loss | Time | Generation Quality |
|---|---|---|---|---|---|---|
| MLP Tanh | 335K | 8 | 5000 | 0.27 | 40s (CPU) | gibberish |
| MLP GELU | 335K | 8 | 5000 | 0.30 | 38s (CPU) | gibberish |
| MLP GELU | 1.1M | 32 | 5000 | 0.21 | 19s (GPU) | structure correct, words wrong |
| Transformer 4L-64D | 209K | 32 | 5000 | 1.65 | 174s | real words, broken grammar |
| Transformer 6L-128D | 1.2M | 32 | 10000 | 1.37 | 1272s | sentences, multi-speaker |
| + LR schedule | 1.2M | 32 | 10000 | 1.37 | 406s | same quality, 3x faster |
| + Block size 128 | 1.2M | 128 | 10000 | 1.32 | 660s | coherent dialogue |
| + AMP | 1.2M | 128 | 10000 | 1.34 | 555s | same, 16% faster |
| GPT-8L-256D + AMP | 6.4M | 128 | 10000 | **0.94** | 1928s | iambic pentameter |

**Perplexity at final model:** `exp(0.94) = 2.56` — on average choosing between ~2.56 equally likely next characters.

---

## Key Concepts — One-Liner Reference

| Concept | What it is |
|---|---|
| Tokenization | Converting text to integers via a character→index dictionary |
| Embedding | Learnable lookup table — each token ID maps to a dense vector |
| Positional embedding | Learned position vectors added to token embeddings so the model knows token order |
| Attention | Each token computes how much to attend to every other token via Q·K dot products |
| Causal mask | Lower triangular matrix preventing tokens from attending to future positions |
| Scaled dot-product | Divide attention scores by √head_size to prevent softmax saturation |
| Multi-head attention | Run N attention heads in parallel, each learning different relationship types |
| Residual connection | `x = x + layer(x)` — preserves original signal, enables deep networks |
| LayerNorm | Normalize each token vector to mean=0, std=1 — stabilizes activations |
| Feedforward block | Per-token MLP with 4x expansion — where the model "thinks" after attention |
| Cross-entropy loss | `-log(probability assigned to correct next token)` — the training signal |
| AdamW | Adaptive per-weight learning rates + weight decay — standard LLM optimizer |
| Gradient clipping | Cap total gradient norm at 1.0 — prevents single bad batch destroying training |
| LR warmup | Ramp learning rate from 0 to max over first N steps — prevents early instability |
| Cosine decay | Gradually reduce LR following cosine curve — smooth convergence at end |
| Mixed precision | Forward in float16, backward in float32 — 40% less memory, faster on GPU |
| Gradient scaling | Multiply loss before backward to prevent float16 underflow of tiny gradients |
| Validation loss | Loss on held-out data — measures generalization, catches overfitting |
| Temperature sampling | Divide logits by T before softmax — lower T = more confident, higher = more random |
| Teacher forcing | During training, feed ground truth context rather than model's own predictions |
| Perplexity | `exp(val_loss)` — average number of equally likely choices the model sees |

---

## Generated Samples — Final Model (GPT-8L-256D)

**Prompt: `ROMEO:`**
```
ROMEO:
Good night, I might be offenced me.
Me ready Carthan go to Petruchio!
Come our hopings to enjoy, upon thy face,
That sees thus spoke them wounds, that she doth her
The seal of heavenly stubble that h
```

**Prompt: `KING RICHARD:`**
```
KING RICHARD:
Somerset, in sail'd by their lawful king,
And made by the time to part the lame.
The noble and take her gone,
And there recomes a holy commanding tremble,
To the princes of those woman, this news nev
```

**Prompt: `First Citizen:`**
```
First Citizen:
Your partine is some part of my brother;
But I by this apprehension.
This gentleman, let me that hath that never doers,
To sweet sorrow to be tide 'gainst God's majesty!

JULIET:
It is, my lord.
```

---

## Training Curves

![Tanh vs GELU](tanh_vs_gelu.png)
![MLP Train vs Val](mlp_train_val.png)
![Transformer Train vs Val](transformer_train_val_amp.png)

---

## How to Run

**Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib numpy
```

**Download data:**
```bash
curl -o data/input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**Run MLP experiment (Tanh vs GELU):**
```bash
python train.py
```

**Run transformer (6L-128D):**
```bash
python transformer/train.py
```

**Run transformer (8L-256D with AMP):**
```bash
python transformer/train1.py
```

---

## What I Learned

Building this from scratch rather than using HuggingFace forced genuine understanding at every level:

- Why attention solves what MLPs cannot — positional relationships between tokens
- Why GELU outperforms Tanh at scale — gradient saturation is a real problem, not a textbook one
- Why val loss and generation quality are not the same metric — MLP at loss 0.21 generated gibberish, transformer at 1.32 generated coherent dialogue
- Why LR scheduling matters — same final loss in 3x less time
- Why residual connections enable depth — without them gradients vanish through many layers
- How mixed precision works mechanically — not just "it's faster" but why float16 needs gradient scaling

Every component was debugged, measured, and compared against a baseline. The transformer at 6.4M params with block_size=128 generates text with correct archaic grammar, real Shakespeare character names, multi-speaker dialogue structure, and near-iambic rhythm — trained from raw character sequences with no linguistic knowledge built in.


