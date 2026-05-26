#!/usr/bin/env python3
"""Train a tiny GPT-style character model and save a reproducible evidence pack.

This script is intentionally small enough for a CPU smoke test, but the course
acceptance target is a real CUDA run. It creates:

- environment_report.json
- training_log.csv
- mini_gpt2_checkpoint.pt
- sample.txt
- README.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TEXT = """
Open-source language models turn tokens into vectors, mix context with causal
attention, and learn to predict the next token. A tiny training run does not
create a useful assistant, but it proves the real loop: data, tokenizer,
embedding, attention, loss, optimizer, checkpoint, generation, and evidence.
When renting a GPU, write down the device, the command, the loss curve, the
sample output, and the stop time before calling the experiment complete.
"""


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, channels = x.shape
        q, k, v = self.qkv(x).split(channels, dim=2)
        q = q.view(batch, steps, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(batch, steps, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(batch, steps, self.n_head, self.head_size).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.mask[:, :, :steps, :steps] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(batch, steps, channels)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, steps = idx.shape
        positions = torch.arange(steps, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def prepare_data(text: str, block_size: int) -> tuple[torch.Tensor, dict[str, int], dict[int, str]]:
    cleaned = "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())
    if not cleaned:
        cleaned = DEFAULT_TEXT.strip()
    while len(cleaned) < block_size + 2:
        cleaned = f"{cleaned}\n{cleaned}"

    chars = sorted(set(cleaned))
    stoi = {ch: index for index, ch in enumerate(chars)}
    itos = {index: ch for ch, index in stoi.items()}
    data = torch.tensor([stoi[ch] for ch in cleaned], dtype=torch.long)
    return data, stoi, itos


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    block_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts]).to(device)
    return x, y


def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    return "".join(itos[int(index)] for index in ids)


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="openllm_gpu_training_run")
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sample-tokens", type=int, default=180)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = choose_device(args.device)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device == "mps" and not (
        getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is False")

    text = args.text_file.read_text(encoding="utf-8") if args.text_file else DEFAULT_TEXT
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, stoi, itos = prepare_data(text, args.block_size)
    config = GPTConfig(
        vocab_size=len(stoi),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    steps = args.steps if args.steps is not None else (500 if device == "cuda" else 80)
    batch_size = args.batch_size if args.batch_size is not None else (64 if device == "cuda" else 16)
    parameter_count = count_parameters(model)
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "mps_available": getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available(),
        "parameter_count": parameter_count,
        "config": asdict(config),
        "steps": steps,
        "batch_size": batch_size,
        "learning_rate": args.lr,
    }
    write_json(output_dir / "environment_report.json", environment)

    log_path = output_dir / "training_log.csv"
    start_time = time.time()
    log_rows: list[dict[str, float | int | str]] = []
    print(f"device: {device}")
    print(f"cuda_name: {environment['cuda_name'] or 'not available'}")
    print(f"parameters: {parameter_count}")

    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "loss", "elapsed_seconds", "device"])
        writer.writeheader()
        for step in range(1, steps + 1):
            model.train()
            x, y = get_batch(data, batch_size, config.block_size, device)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("loss was not computed")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step == 1 or step % 50 == 0 or step == steps:
                elapsed = time.time() - start_time
                row = {
                    "step": step,
                    "loss": round(float(loss.item()), 6),
                    "elapsed_seconds": round(elapsed, 3),
                    "device": device,
                }
                writer.writerow(row)
                file.flush()
                log_rows.append(row)
                print(
                    f"step {step:04d} | loss {row['loss']:.4f} | "
                    f"elapsed {row['elapsed_seconds']:.1f}s"
                )

    checkpoint = {
        "model_state": model.state_dict(),
        "config": asdict(config),
        "stoi": stoi,
        "itos": itos,
        "environment": environment,
    }
    checkpoint_path = output_dir / "mini_gpt2_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    prompt_id = torch.tensor([[stoi[DEFAULT_TEXT.strip()[0]]]], dtype=torch.long, device=device)
    sample = decode(model.generate(prompt_id, args.sample_tokens)[0].cpu(), itos)
    (output_dir / "sample.txt").write_text(sample + "\n", encoding="utf-8")

    first_loss = log_rows[0]["loss"] if log_rows else None
    last_loss = log_rows[-1]["loss"] if log_rows else None
    readme = f"""# Mini GPT-2 Training Evidence

## Command

```bash
python mini_gpt2_train.py --output-dir {output_dir}
```

## Environment

- device: {device}
- cuda_name: {environment['cuda_name'] or 'not available'}
- torch: {torch.__version__}
- parameters: {parameter_count}

## Training

- steps: {steps}
- batch_size: {batch_size}
- first_logged_loss: {first_loss}
- last_logged_loss: {last_loss}

## Files

- environment_report.json
- training_log.csv
- mini_gpt2_checkpoint.pt
- sample.txt

## Acceptance Note

CPU or MPS output is a smoke test. A final course pass should include a run
where `environment_report.json` says `"device": "cuda"` and `training_log.csv`
has at least three logged loss rows.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"checkpoint: {checkpoint_path}")
    print(f"training_log: {log_path}")
    print("--- sample ---")
    print(sample)


if __name__ == "__main__":
    main()
