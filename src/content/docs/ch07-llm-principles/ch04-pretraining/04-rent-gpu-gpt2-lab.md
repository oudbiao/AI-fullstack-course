---
title: "7.4.5 Rent a GPU and Run a Hand-Built GPT-2"
description: "Choose free or low-cost GPU compute, set up PyTorch, run a single-file mini GPT-2 trainer, and understand tokenizer, attention, loss, and generation line by line."
sidebar:
  order: 15
head:
  - tag: meta
    attrs:
      name: keywords
      content: "rent GPU, hand-built GPT-2, mini GPT, PyTorch, Kaggle, Colab, AutoDL, RunPod"
---
![Hand-built GPT-2 lab whiteboard: free notebook, rented GPU, environment check, mini GPT-2 training, qualitative loss trend, sample generation, and shutdown evidence.](/img/course/ch07-mini-gpt2-lab-whiteboard-en.webp)

:::tip[Where This Lesson Fits]
This lesson puts the Transformer and pretraining objective onto a real machine.

You do not need to train the full GPT-2 124M model. The goal is to run a mini GPT-2, see the loss decrease, generate a short sample, and explain what each part of the code does.
:::

## Learning Goals

- Decide when to use a free notebook and when to rent a low-cost GPU.
- Create a Python, PyTorch, and CUDA-ready training environment.
- Run a single-file mini GPT-2 training script.
- Explain embedding, causal self-attention, MLP, loss, and generation.
- Save training logs, hardware information, and sample output as evidence.

---

## 1. Choose a GPU Option First

Do not start by chasing the largest card. Course labs should first make sure every learner can finish.

| Option | Best for | Recommended use | Watch out for |
|---|---|---|---|
| Kaggle Notebook | Free-first public courses | Enable GPU and run mini GPT-2 | Quotas change and GPU is not guaranteed |
| Colab Free | Fast trial runs | Validate code and logs | GPU model and session length vary |
| Lightning AI free tier | Cloud development workflow | Save projects and repeat experiments | Free credits can run out |
| AutoDL / RunPod | Stable 1-3 hour labs | Rent RTX 4090, L4, A10, or A5000 | Stop and delete instances when done |
| A100 / H100 | Understanding large-scale costs | Demo or advanced challenge only | Too expensive for required student work |

### Recommended Config for This Lesson

| Goal | Minimum | More comfortable |
|---|---|---|
| Run the script | CPU or free T4 | T4, L4, A10 |
| See clear loss decrease | Free T4 for 300-800 steps | 4090 or A5000 for 1000-3000 steps |
| Try a larger model | 16GB VRAM | 24GB VRAM |

The default script is tiny. It can run on CPU, just more slowly. Renting a GPU mainly shortens the wait and teaches the real training workflow.

---

## 2. Checklist Before Paying

Before you start a paid machine, confirm four things:

1. Budget: decide the maximum cost for this lab, such as a few dollars.
2. Machine: prefer 16GB or 24GB VRAM; the most expensive card is unnecessary.
3. Image: choose a PyTorch image, ideally with CUDA preinstalled.
4. Exit path: know where to stop billing and delete the instance.

Common routes:

```text
Free route: Kaggle / Colab -> enable GPU -> upload or create script -> run
China low-cost route: AutoDL -> choose PyTorch image -> open Jupyter or SSH -> run
International low-cost route: RunPod -> choose PyTorch template -> open terminal -> run
```

Cost rule: run 50 steps in a free notebook first, then rent a GPU for a longer run. Do not burn money while debugging the environment.

---

## 3. Open the Environment and Check PyTorch

Run this in a notebook or remote terminal:

```bash
python -V
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
```

Expected output is similar to:

```text
torch: 2.x.x
cuda available: True
device: Tesla T4
```

If `cuda available` is `False`, do not start training yet. Check whether the notebook accelerator is enabled or whether the cloud instance uses a CUDA PyTorch image.

---

## 4. Create the Single-File Script

Create `mini_gpt2_train.py`. Copy the full script first; do not tune parameters before the first successful run.

```python
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


text = """
To build a language model, we ask it to predict the next token.
The model reads previous tokens, mixes context with attention, and produces logits.
Small experiments teach the same training loop as large models.
"""


chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)


def decode(ids):
    return "".join(itos[int(i)] for i in ids)


def get_batch(batch_size, block_size, device):
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    config = GPTConfig(vocab_size=len(chars))
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    steps = 500 if device == "cuda" else 120
    batch_size = 64 if device == "cuda" else 16
    print("device:", device)
    print("parameters:", sum(p.numel() for p in model.parameters()))

    start_time = time.time()
    for step in range(1, steps + 1):
        x, y = get_batch(batch_size, config.block_size, device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0:
            elapsed = time.time() - start_time
            print(f"step {step:04d} | loss {loss.item():.4f} | elapsed {elapsed:.1f}s")

    prompt = torch.tensor([[stoi["T"]]], dtype=torch.long, device=device)
    generated = model.generate(prompt, max_new_tokens=180)[0].cpu()
    print("\n--- sample ---")
    print(decode(generated))


if __name__ == "__main__":
    main()
```

Run it:

```bash
python mini_gpt2_train.py | tee mini_gpt2_train_log.txt
```

Expected output:

```text
device: cuda
parameters: about 100k
step 0001 | loss 3.5832 | elapsed 0.2s
step 0050 | loss 3.1120 | elapsed 1.6s
...
--- sample ---
To build a language model...
```

The generated text does not need to be elegant. If loss decreases and the model emits characters, the training loop works.

---

## 5. Line-by-Line Explanation

### Text and tokenizer

```python
chars = sorted(set(text))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
```

This is a character-level tokenizer. Real GPT-2 uses BPE tokens, but characters keep the lab dependency-free and focus attention on the model.

### Next-token batch

```python
x = data[i : i + block_size]
y = data[i + 1 : i + block_size + 1]
```

`x` is the input and `y` is the answer. The model reads token 0 through token T-1 and predicts token 1 through token T.

### Config object

```python
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
```

These values control model size: context length, number of blocks, attention heads, and embedding width.

### QKV and multi-head attention

```python
q, k, v = self.qkv(x).split(C, dim=2)
q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
```

`x` has shape `[B, T, C]`. After the split and reshape, each head receives its own `[B, head, T, head_size]` view.

### Causal mask

```python
mask = torch.tril(torch.ones(config.block_size, config.block_size))
scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
```

The lower-triangular mask prevents each position from seeing future tokens. This is the core rule behind next-token prediction in decoder-only models.

### Transformer block

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

Attention mixes context. The MLP transforms each position. Residual connections keep information and gradients moving.

### Embeddings

```python
x = self.token_emb(idx) + self.pos_emb(positions)
```

Token embedding says what the token is. Position embedding says where it appears.

### Logits and loss

```python
logits = self.lm_head(x)
loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
```

`logits` has shape `[B, T, vocab_size]`. Cross entropy rewards the model for assigning higher probability to the true next token.

### Training loop

```python
logits, loss = model(x, y)
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

This is the core loop: forward, clear gradients, backpropagate, update parameters.

### Generate

```python
logits = logits[:, -1, :]
probs = F.softmax(logits, dim=-1)
next_id = torch.multinomial(probs, num_samples=1)
```

Generation reads the last position, samples the next token, appends it, and repeats.

---

## 6. If You Really Rent a GPU

### Kaggle or Colab

1. Create a notebook.
2. Enable GPU in settings.
3. Run the PyTorch check and confirm `cuda available: True`.
4. Create `mini_gpt2_train.py`.
5. Run `python mini_gpt2_train.py | tee mini_gpt2_train_log.txt`.
6. Save hardware info, loss lines, and the generated sample.

### AutoDL or RunPod

1. Choose a PyTorch image.
2. Choose a 16GB or 24GB VRAM machine.
3. Open JupyterLab or SSH terminal.
4. Run the PyTorch check.
5. Save the script and train.
6. Stop the instance immediately after the run and confirm billing has stopped.

---

## Common Issues

| Symptom | Likely cause | Fix |
|---|---|---|
| `cuda available: False` | GPU disabled or wrong image | Enable accelerator or rebuild with CUDA/PyTorch image |
| `CUDA out of memory` | Batch, context, or model too large | Reduce `batch_size`, then `block_size` or `n_embd` |
| Loss does not decrease | Too few steps, data too short, bad LR | Run 500 steps before judging trend |
| Generated text is messy | Model and data are tiny | Normal for this lab; mechanism is the goal |
| Billing continues | Instance still running | Stop the instance and verify in the console |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
platform_choice: Kaggle/Colab/Lightning/AutoDL/RunPod
hardware_info: torch version, CUDA status, GPU model
training_log: at least three lines with step, loss, elapsed
code_location: identify embedding, attention, loss, and generate in the script
cost_record: if rented, record runtime and cost, then confirm shutdown
```

## Pass Check

You pass when `mini_gpt2_train.py` runs on CPU, free GPU, or rented GPU, `mini_gpt2_train_log.txt` is saved, and you can explain how input tokens pass through embedding, attention, MLP, lm head, and cross entropy to learn next-token prediction.

<details>
<summary>Check reasoning and explanation</summary>

1. The goal is not beautiful generated text. The goal is to run the full path.
2. A passing log includes hardware info, parameter count, several loss lines, and one sample.
3. If you rented a GPU, your evidence must state that the instance was stopped.
4. CPU completion also counts. GPU is a speed and environment lesson, not a requirement.

</details>
