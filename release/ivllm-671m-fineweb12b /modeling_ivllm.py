"""Self-contained IvLLM 671M model definition.

Usage:
    import torch, json
    from safetensors.torch import load_file
    from modeling_ivllm import IvLLM

    model = IvLLM()
    sd = load_file('model.safetensors')
    model.load_state_dict(sd, strict=True)
    model.eval()
"""

import os
import glob
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from einops import rearrange, repeat


# =============================================================================
# 2. HYPERPARAMETERS & STREAMING CONFIGURATION
# =============================================================================
# Model Architecture (671M Parameters)
VOCAB_SIZE = 50304         
DIM = 1536                 
NUM_BLOCKS = 24            
NUM_Q_HEADS = 24           
NUM_KV_GROUPS = 6          
QUERIES_PER_GROUP = NUM_Q_HEADS // NUM_KV_GROUPS
HEAD_DIM = DIM // NUM_Q_HEADS 

# Batch Sizing
SEQ_LENGTH = 2048          
MICRO_BATCH_SIZE = 8     
GLOBAL_BATCH_SIZE = 480    
assert GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * ddp_world_size) == 0
GRAD_ACCUM_STEPS = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * ddp_world_size)

# Training Horizon (FineWeb10B GPT-2 pretokenized shards)
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQ_LENGTH
NUM_EPOCHS = 1                  # Passes over the downloaded train shards
CHECKPOINT_INTERVAL = 500       # Steps between val + checkpoint backups

# Learning Rate
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.1

# FineWeb10B shards downloaded via the kjj0/fineweb10B-gpt2 script
# (train: fineweb_train_NNNNNN.bin, val: fineweb_val_000000.bin)
DATA_DIR = "data"
CHECKPOINT_DIR = "checkpoints"
if master_process:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# nanoGPT/FineWeb .bin format: 256 int32 header then uint16 tokens
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4   # 1024 bytes
MAGIC_NUMBER = 20240520
VERSION = 1

# =============================================================================
# 3. CORE ARCHITECTURE MODULES
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep computation entirely in float32, only downcast at the very end
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq) 
        freqs = torch.cat((freqs, freqs), dim=-1) 
        self.register_buffer("cos", torch.cos(freqs), persistent=False)
        self.register_buffer("sin", torch.sin(freqs), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[2]
        cos = self.cos[:seq_len].view(1, 1, seq_len, -1)
        sin = self.sin[:seq_len].view(1, 1, seq_len, -1)
        
        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
            
        q_out = (q * cos) + (rotate_half(q) * sin)
        k_out = (k * cos) + (rotate_half(k) * sin)
        return q_out, k_out

class GroupedQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(DIM, NUM_Q_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(NUM_Q_HEADS * HEAD_DIM, DIM, bias=False)
        self.wo.NANOGPT_SCALE_INIT = 1 

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        q = rearrange(self.wq(x), 'b t (h d) -> b h t d', h=NUM_Q_HEADS)
        k = rearrange(self.wk(x), 'b t (g d) -> b g t d', g=NUM_KV_GROUPS)
        v = rearrange(self.wv(x), 'b t (g d) -> b g t d', g=NUM_KV_GROUPS)

        q, k = rope(q, k)

        k = repeat(k, 'b g t d -> b (g r) t d', r=QUERIES_PER_GROUP)
        v = repeat(v, 'b g t d -> b (g r) t d', r=QUERIES_PER_GROUP)

        context = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        context = rearrange(context, 'b h t d -> b t (h d)')
        return self.wo(context)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = int(2 * (4 * DIM / 3))
        hidden_dim = ((hidden_dim + 127) // 128) * 128 
        self.w_gate_val = nn.Linear(DIM, 2 * hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, DIM, bias=False)
        self.w_down.NANOGPT_SCALE_INIT = 1 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = self.w_gate_val(x)
        gate, val = rearrange(combined, 'b t (split h) -> split b t h', split=2)
        return self.w_down(F.silu(gate) * val)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = RMSNorm(DIM)
        self.attn = GroupedQueryAttention()
        self.ffn_norm = RMSNorm(DIM)
        self.ffn = SwiGLU()

    def forward(self, x: torch.Tensor, rope: RoPE) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), rope)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class IvLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, DIM)
        self.rope = RoPE(dim=HEAD_DIM, max_seq_len=SEQ_LENGTH)
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(NUM_BLOCKS)])
        self.final_norm = RMSNorm(DIM)
        self.output_layer = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        
        # Weight Tying
        self.embeddings.weight = self.output_layer.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * NUM_BLOCKS) ** -0.5
            
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        x = self.embeddings(tokens)
        for block in self.blocks:
            x = block(x, self.rope)
        x = self.final_norm(x)
        logits = self.output_layer(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(rearrange(logits, 'b t v -> (b t) v'), rearrange(targets, 'b t -> (b t)'))
        return logits, loss

# =============================================================================
