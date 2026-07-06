---
license: apache-2.0
language:
- en
tags:
- pretrained
- causal-language-model
- transformer
- fineweb
- pytorch
library_name: pytorch
---

# IvLLM 671M

A **671 million parameter** decoder-only Transformer language model pre-trained from scratch on approximately **12.2 billion tokens** from the FineWeb dataset. The model uses the GPT-2 tokenizer (tiktoken) with a vocabulary size of **50,304**.

This checkpoint is intended as a lightweight foundation model that can be used for continued pre-training, domain adaptation, or downstream fine-tuning. It reaches a validation loss of **2.95** on the FineWeb validation set.

---

## Model Configuration

| Component | Value |
|-----------|-------|
| Parameters | 671M |
| Hidden Dimension | 1536 |
| Transformer Layers | 24 |
| Attention Heads | 24 |
| KV Groups | 6 (Grouped Query Attention) |
| Head Dimension | 64 |
| Maximum Context Length | 2048 |
| Feed Forward Network | SwiGLU (4096 hidden units) |
| Normalization | RMSNorm |
| Positional Encoding | Rotary Embeddings (RoPE, θ = 10,000) |
| Embedding Tying | Enabled |

---

## Repository Contents

- **model.safetensors** – Model weights stored in BF16 format.
- **config.json** – Architecture and training configuration.
- **modeling_ivllm.py** – Standalone PyTorch implementation of the model.

---

## Loading the Model

```python
import torch
import tiktoken
from safetensors.torch import load_file
from modeling_ivllm import IvLLM

model = IvLLM().cuda().eval()
weights = load_file("model.safetensors")
model.load_state_dict(weights)

tokenizer = tiktoken.get_encoding("gpt2")

prompt = "The capital of France is"
tokens = torch.tensor(
    [tokenizer.encode_ordinary(prompt)],
    device="cuda"
)

with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    for _ in range(30):
        logits, _ = model(tokens[:, -2048:])
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens = torch.cat([tokens, next_token], dim=1)

print(tokenizer.decode(tokens[0].tolist()))
```

---

## Training Summary

| Metric | Value |
|--------|-------|
| Dataset | FineWeb |
| Total Tokens | ~12.24 Billion |
| Batch Size | 480 × 2048 tokens |
| Tokens / Step | ~983K |
| Optimizer | AdamW |
| Betas | (0.9, 0.95) |
| Weight Decay | 0.1 |
| Learning Rate | Cosine Decay (3e-4 → 3e-5) |
| Warm-up Steps | 2,000 |
| Hardware | 8 × NVIDIA H100 80GB SXM |
| Throughput | ~760K tokens/sec |
| MFU | ~54% |
| Final Training Loss | 2.94 |
| Final Validation Loss | 2.95 |

---
