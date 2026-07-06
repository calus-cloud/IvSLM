"""Build a clean HF-uploadable release of the legacy 671M ivllm checkpoint."""

import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

WORKSPACE = Path("/home/ubuntu/ivllm")
SRC_CKPT = WORKSPACE / "checkpoints/legacy_pre_qknorm/ivllm_latest.pt"
SRC_BAK = WORKSPACE / "ivllm.py.bak"
OUT_DIR = WORKSPACE / "release/ivllm-671m-fineweb12b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[1/5] Loading {SRC_CKPT} ...")
ckpt = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
sd = ckpt["model_state_dict"]
global_step = int(ckpt.get("global_step", 0))
print(f"      keys: {len(sd)}  global_step: {global_step}")

# Detach + clone so safetensors sees independent storages (handles tied weights).
print("[2/5] Cloning tensors (breaking tied storage for safetensors) ...")
sd_clean = {k: v.detach().to(torch.bfloat16).contiguous().clone() for k, v in sd.items()}
total_params = sum(t.numel() for t in sd_clean.values())
print(f"      total params (bf16): {total_params/1e6:.2f}M  "
      f"({sum(t.numel()*t.element_size() for t in sd_clean.values())/1e9:.2f} GB)")

print("[3/5] Writing model.safetensors ...")
save_file(
    sd_clean,
    str(OUT_DIR / "model.safetensors"),
    metadata={"format": "pt", "trained_tokens": "~12B fineweb", "step": str(global_step)},
)

print("[4/5] Writing config.json ...")
config = {
    "architecture": "IvLLM",
    "vocab_size": 50304,
    "dim": 1536,
    "num_blocks": 24,
    "num_q_heads": 24,
    "num_kv_groups": 6,
    "head_dim": 64,
    "seq_length": 2048,
    "rope_theta": 10000.0,
    "tied_embeddings": True,
    "norm": "rmsnorm",
    "attention": "GQA",
    "ffn": "SwiGLU",
    "tokenizer": "gpt2 (tiktoken)",
    "dtype": "bfloat16",
    "trained_tokens_approx": 12_238_848_000,
    "training_step": global_step,
    "training_data": "FineWeb (kjj0/fineweb100B-gpt2) first ~120 shards",
    "params_total": total_params,
}
(OUT_DIR / "config.json").write_text(json.dumps(config, indent=2))

# Extract only the model classes from ivllm.py.bak into modeling_ivllm.py.
print("[5/5] Writing modeling_ivllm.py ...")
src = SRC_BAK.read_text()
# Keep imports + section 3 (architecture). Strip section 4 onwards.
# The bak ends section 3 right before "# 4. DISTRIBUTED DATALOADER".
m = re.search(r"# 4\. DISTRIBUTED DATALOADER", src)
arch_only = src[: m.start()].rstrip() + "\n"

# Trim DDP boot section (lines 17-30) — not needed for inference.
arch_only = re.sub(
    r"# =+\n# 1\. DDP INITIALIZATION.*?torch\.set_float32_matmul_precision\('high'\)\n",
    "",
    arch_only,
    flags=re.DOTALL,
)
header = (
    "\"\"\"Self-contained IvLLM 671M model definition.\n\n"
    "Usage:\n"
    "    import torch, json\n"
    "    from safetensors.torch import load_file\n"
    "    from modeling_ivllm import IvLLM\n\n"
    "    model = IvLLM()\n"
    "    sd = load_file('model.safetensors')\n"
    "    model.load_state_dict(sd, strict=True)\n"
    "    model.eval()\n"
    "\"\"\"\n\n"
)
(OUT_DIR / "modeling_ivllm.py").write_text(header + arch_only)

readme = f"""---
license: apache-2.0
language:
- en
tags:
- pretrained
- causal-lm
- gpt2-tokenizer
- fineweb
library_name: pytorch
---

# IvLLM 671M (fineweb-12B checkpoint)

A 671 M parameter decoder-only LM trained from scratch on ~12 B tokens of
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
([kjj0/fineweb100B-gpt2](https://huggingface.co/datasets/kjj0/fineweb100B-gpt2) pre-tokenized shards).
GPT-2 (tiktoken) tokenizer; vocab 50 304.

This is an early pretrain checkpoint — useful as a small base model or a starting point for
further training. Final val loss on FineWeb: **2.95**.

## Architecture

| | |
|---|---|
| dim | 1536 |
| blocks | 24 |
| Q heads / KV groups | 24 / 6 (GQA, ratio 4) |
| head_dim | 64 |
| seq_len | 2048 |
| ffn | SwiGLU (hidden 4096) |
| norm | RMSNorm |
| pos | RoPE θ=10 000 |
| tied embeddings | yes |
| params | 671 M |

## Files
- `model.safetensors` — bf16 weights ({total_params/1e6:.0f} M params, tied embed+lm_head saved separately)
- `config.json` — architecture + training summary
- `modeling_ivllm.py` — self-contained PyTorch model definition

## Quick start

```python
import torch
from safetensors.torch import load_file
from modeling_ivllm import IvLLM
import tiktoken

model = IvLLM().to('cuda').eval()
sd = load_file('model.safetensors')
model.load_state_dict(sd, strict=True)

enc = tiktoken.get_encoding('gpt2')
ids = torch.tensor([enc.encode_ordinary("The capital of France is")], device='cuda')

# Greedy decoding for 30 tokens
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    for _ in range(30):
        logits, _ = model(ids[:, -2048:])
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
print(enc.decode(ids[0].tolist()))
```

## Training details

| | |
|---|---|
| Tokens seen | ~12.24 B |
| Global batch | 480 × 2048 ≈ 983 k tokens/step |
| Optimizer | AdamW (β=0.9/0.95, wd=0.1, fused) |
| LR schedule | cosine, peak 3e-4 → min 3e-5, 2 000 step warmup |
| Hardware | 8 × H100 80 GB SXM |
| Steady-state | ~760 k tok/s, ~54 % MFU |
| Final train loss | 2.94 |
| Final val loss | 2.95 |

Trained as part of the [ivllm](https://github.com/) workspace. Subsequent training runs use a
larger 1.17 B architecture with QK-Norm, value-residual learning, per-head sigmoid gating, and
output z-loss — published separately.
"""
(OUT_DIR / "README.md").write_text(readme)

print("\nDone.")
for p in sorted(OUT_DIR.glob("*")):
    sz = p.stat().st_size
    print(f"  {p.name:30s}  {sz/1e6:8.2f} MB")
