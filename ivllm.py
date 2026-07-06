import os
import glob
import time
import math
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from contextlib import nullcontext
from einops import rearrange

# =============================================================================
# 1. DDP INITIALIZATION & HARDWARE MAPPING
# =============================================================================
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if master_process:
        print("WARNING: Running on a single GPU. Use torchrun for multi-GPU.")

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# =============================================================================
# 2. HYPERPARAMETERS & STREAMING CONFIGURATION
# =============================================================================
# Model Architecture (~1.17B Parameters): DIM=2048, head_dim=128 (tensor-core friendly).
# QK-Norm + Value Residual + per-head output gating + z-loss for modern stability.
VOCAB_SIZE = 50304
DIM = 2048
NUM_BLOCKS = 24
NUM_Q_HEADS = 16
NUM_KV_GROUPS = 4
QUERIES_PER_GROUP = NUM_Q_HEADS // NUM_KV_GROUPS
HEAD_DIM = DIM // NUM_Q_HEADS  # 128

# Batch Sizing — picked so 8xH100 80GB stays well-utilized and divides cleanly.
SEQ_LENGTH = 2048
MICRO_BATCH_SIZE = int(os.environ.get("IVLLM_MICRO_BATCH", 10))
GLOBAL_BATCH_SIZE = int(os.environ.get("IVLLM_GLOBAL_BATCH", 480))
assert GLOBAL_BATCH_SIZE % (MICRO_BATCH_SIZE * ddp_world_size) == 0, (
    f"GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE} not divisible by "
    f"MICRO_BATCH_SIZE*world_size={MICRO_BATCH_SIZE*ddp_world_size}"
)
GRAD_ACCUM_STEPS = GLOBAL_BATCH_SIZE // (MICRO_BATCH_SIZE * ddp_world_size)
TOKENS_PER_STEP = GLOBAL_BATCH_SIZE * SEQ_LENGTH

NUM_EPOCHS = 1
CHECKPOINT_INTERVAL = 500

# ---- Learning-rate schedule ----
# Original (legacy) cosine config — retained for compatibility.
MAX_LR = 3e-4
MIN_LR = 3e-5
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.1

# New WSD / cosine schedule with re-warmup support.
# IVLLM_SCHEDULE = 'wsd' (default) | 'cosine' | 'legacy_cosine'.
SCHEDULE = os.environ.get("IVLLM_SCHEDULE", "wsd").lower()
PEAK_LR = float(os.environ.get("IVLLM_PEAK_LR", 1.5e-4))
FLOOR_LR = float(os.environ.get("IVLLM_FLOOR_LR", 3e-6))
RESUME_LR = float(os.environ.get("IVLLM_RESUME_LR", MIN_LR))  # LR to ramp up from
REWARMUP_STEPS = int(os.environ.get("IVLLM_REWARMUP_STEPS", 500))
COOLDOWN_FRAC = float(os.environ.get("IVLLM_COOLDOWN_FRAC", 0.2))

# ---- Run horizon ----
# Explicit override (highest priority): exact step count.
_ENV_MAX_STEPS = os.environ.get("IVLLM_MAX_STEPS")
# Or token budget for the *full* pretrain (e.g. '100B', '50B').
_ENV_TARGET_TOKENS = os.environ.get("IVLLM_TARGET_TOKENS")


def _parse_token_count(s):
    s = str(s).strip().replace("_", "")
    mult = 1
    if s[-1] in "Kk":
        mult, s = 1_000, s[:-1]
    elif s[-1] in "Mm":
        mult, s = 1_000_000, s[:-1]
    elif s[-1] in "Bb":
        mult, s = 1_000_000_000, s[:-1]
    elif s[-1] in "Tt":
        mult, s = 1_000_000_000_000, s[:-1]
    return int(float(s) * mult)


DATA_DIR = os.environ.get("IVLLM_DATA_DIR", "fineweb100B")
CHECKPOINT_DIR = "checkpoints"
if master_process:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---- Data mix (weighted shard sampling) ----
# IVLLM_MIX = "fineweb_train=0.88,math_train=0.06,code_train=0.06"
_ENV_MIX = os.environ.get("IVLLM_MIX", "")


def _parse_mix(s):
    if not s:
        return None
    out = {}
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        k, v = piece.split("=")
        k = k.strip()
        if not k.endswith("_"):
            k = k + "_"
        out[k] = float(v)
    return out or None


DATA_MIX = _parse_mix(_ENV_MIX)

# Logging
LOG_INTERVAL = int(os.environ.get("IVLLM_LOG_INTERVAL", 10))
WANDB_ENABLED = os.environ.get("IVLLM_WANDB", "0") == "1"
WANDB_PROJECT = os.environ.get("IVLLM_WANDB_PROJECT", "ivllm")
WANDB_RUN_NAME = os.environ.get("IVLLM_WANDB_RUN", None)

# nanoGPT/FineWeb .bin format: 256 int32 header then uint16 tokens
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
MAGIC_NUMBER = 20240520
VERSION = 1

# Train shards may come from any of these prefixes (fineweb + math + code mixes)
TRAIN_PREFIXES = ("fineweb_train_", "math_train_", "code_train_")
VAL_PREFIX = "fineweb_val_"

# Output softmax z-loss (PaLM/OLMo-style). 0 to disable.
Z_LOSS_WEIGHT = float(os.environ.get("IVLLM_Z_LOSS", 1e-4))

# ---- In-training evaluation (lm-eval-harness) ----
# Quick subset every IVLLM_EVAL_INTERVAL_STEPS, full eval every
# IVLLM_EVAL_FULL_INTERVAL_TOKENS. Both run on rank 0 only; other ranks barrier.
EVAL_ENABLED = os.environ.get("IVLLM_EVAL", "1") == "1"
EVAL_INTERVAL_STEPS = int(os.environ.get("IVLLM_EVAL_INTERVAL_STEPS", 500))
EVAL_FULL_INTERVAL_TOKENS = os.environ.get("IVLLM_EVAL_FULL_INTERVAL_TOKENS", "2B")
EVAL_QUICK_TASKS = os.environ.get(
    "IVLLM_EVAL_QUICK_TASKS", "hellaswag,piqa"
)
EVAL_QUICK_LIMIT = int(os.environ.get("IVLLM_EVAL_QUICK_LIMIT", 200))
EVAL_FULL_TASKS = os.environ.get(
    "IVLLM_EVAL_FULL_TASKS",
    "hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai",
)
EVAL_BATCH_SIZE = int(os.environ.get("IVLLM_EVAL_BATCH_SIZE", 16))

# =============================================================================
# 3. CORE ARCHITECTURE MODULES (unchanged)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.wq = nn.Linear(DIM, NUM_Q_HEADS * HEAD_DIM, bias=False)
        self.wk = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wv = nn.Linear(DIM, NUM_KV_GROUPS * HEAD_DIM, bias=False)
        self.wo = nn.Linear(NUM_Q_HEADS * HEAD_DIM, DIM, bias=False)
        self.wo.NANOGPT_SCALE_INIT = 1
        # QK-Norm (OLMo-2 style): per-head RMSNorm on Q and K before RoPE/SDPA.
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        # Per-head sigmoid output gate (DeepSeek-V3 / NSA style).
        # One scalar per head per token; init to zero → gate=0.5 at start.
        self.head_gate = nn.Linear(DIM, NUM_Q_HEADS, bias=False)
        self.head_gate.HEAD_GATE_INIT = 1
        # Value Residual Learning: layer>0 mixes its V with the first layer's V
        # via a learned scalar gate. sigmoid(0)=0.5 at init.
        if layer_idx > 0:
            self.v_lambda = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("v_lambda", None)

    def forward(self, x: torch.Tensor, rope: RoPE,
                v_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        q = rearrange(self.wq(x), 'b t (h d) -> b h t d', h=NUM_Q_HEADS)
        k = rearrange(self.wk(x), 'b t (g d) -> b g t d', g=NUM_KV_GROUPS)
        v = rearrange(self.wv(x), 'b t (g d) -> b g t d', g=NUM_KV_GROUPS)

        # QK-Norm over head_dim (static branch per layer for compile).
        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = rope(q, k)

        if self.layer_idx == 0:
            v_out = v
            v_first_out = v
        else:
            gate_v = torch.sigmoid(self.v_lambda)
            v_out = gate_v * v_first + (1.0 - gate_v) * v
            v_first_out = v_first

        context = F.scaled_dot_product_attention(q, k, v_out, is_causal=True, enable_gqa=True)
        # Per-head sigmoid gate: down/up-weight heads based on token content.
        head_g = torch.sigmoid(self.head_gate(x))  # (B, T, H)
        context = context * rearrange(head_g, 'b t h -> b h t 1')
        context = rearrange(context, 'b h t d -> b t (h d)')
        return self.wo(context), v_first_out

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
    def __init__(self, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(DIM)
        self.attn = GroupedQueryAttention(layer_idx)
        self.ffn_norm = RMSNorm(DIM)
        self.ffn = SwiGLU()

    def forward(self, x: torch.Tensor, rope: RoPE,
                v_first: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, v_first = self.attn(self.attn_norm(x), rope, v_first)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, v_first

class IvLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(VOCAB_SIZE, DIM)
        self.rope = RoPE(dim=HEAD_DIM, max_seq_len=SEQ_LENGTH)
        self.blocks = nn.ModuleList([TransformerBlock(i) for i in range(NUM_BLOCKS)])
        self.final_norm = RMSNorm(DIM)
        self.output_layer = nn.Linear(DIM, VOCAB_SIZE, bias=False)
        self.embeddings.weight = self.output_layer.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):
            std *= (2 * NUM_BLOCKS) ** -0.5
        if hasattr(module, 'HEAD_GATE_INIT'):
            # Zero-init the head gate so sigmoid(0)=0.5 → neutral start.
            std = 0.0
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(self, tokens: torch.Tensor, targets: torch.Tensor = None):
        x = self.embeddings(tokens)
        v_first: torch.Tensor | None = None
        for block in self.blocks:
            x, v_first = block(x, self.rope, v_first)
        x = self.final_norm(x)
        logits = self.output_layer(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                rearrange(logits, 'b t v -> (b t) v'),
                rearrange(targets, 'b t -> (b t)'),
            )
            if Z_LOSS_WEIGHT > 0:
                log_z = torch.logsumexp(logits.float(), dim=-1)
                loss = ce + Z_LOSS_WEIGHT * (log_z ** 2).mean()
            else:
                loss = ce
        return logits, loss

# =============================================================================
# 4. DISTRIBUTED DATALOADER
#    - Pools many shard prefixes (fineweb + math + code)
#    - Rescans the data dir on every shard rollover (picks up new downloads)
#    - Returns pinned-memory CPU tensors; the prefetcher copies to GPU async
# =============================================================================

def _read_shard_header(file_path: str) -> int:
    header = np.fromfile(file_path, dtype=np.int32, count=HEADER_INTS)
    if header.shape[0] < 3:
        raise ValueError(f"{file_path}: truncated header")
    if int(header[0]) != MAGIC_NUMBER:
        raise ValueError(f"{file_path}: bad magic {int(header[0])}, expected {MAGIC_NUMBER}")
    if int(header[1]) != VERSION:
        raise ValueError(f"{file_path}: unsupported version {int(header[1])}")
    return int(header[2])

def _scan_shards(data_dir: str, prefixes) -> list[str]:
    files = []
    for prefix in prefixes:
        files.extend(glob.glob(os.path.join(data_dir, f"{prefix}*.bin")))
    return sorted(files)

class DistributedShardLoader:
    def __init__(self, data_dir: str, prefixes, batch_size: int, seq_len: int,
                 rank: int, world_size: int, is_train: bool = True,
                 mix: dict | None = None):
        self.data_dir = data_dir
        self.prefixes = tuple(prefixes) if not isinstance(prefixes, str) else (prefixes,)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.is_train = is_train
        # Optional {prefix: weight} mapping for weighted-shard sampling.
        self.mix = mix
        self.shard_step = 0  # increments on every shard rollover; used as RNG seed

        self.files = _scan_shards(data_dir, self.prefixes)
        if not self.files:
            raise FileNotFoundError(
                f"No shards matching {self.prefixes} in '{data_dir}'."
            )

        self.total_tokens = sum(_read_shard_header(f) for f in self.files)
        self.current_shard_idx = 0
        self.position = self.rank * self.batch_size * self.seq_len
        self.load_shard()

    def load_shard(self):
        file_path = self.files[self.current_shard_idx]
        ntokens = _read_shard_header(file_path)
        self.mmap = np.memmap(file_path, dtype=np.uint16, mode='r',
                              offset=HEADER_BYTES, shape=(ntokens,))

    def _refresh_files(self):
        """Pick up newly-arrived shards. Only meaningful for the train pool."""
        if not self.is_train:
            return
        new_files = _scan_shards(self.data_dir, self.prefixes)
        if len(new_files) != len(self.files):
            # Stay anchored at the current shard regardless of new additions.
            current_name = self.files[self.current_shard_idx]
            self.files = new_files
            try:
                self.current_shard_idx = self.files.index(current_name)
            except ValueError:
                self.current_shard_idx = self.current_shard_idx % len(self.files)

    def state_dict(self):
        return {
            'current_shard_idx': self.current_shard_idx,
            'position': self.position,
            'file_name': self.files[self.current_shard_idx],
            'shard_step': self.shard_step,
        }

    def load_state_dict(self, state):
        # Re-anchor by filename so a growing pool still resumes correctly.
        if 'file_name' in state and state['file_name'] in self.files:
            self.current_shard_idx = self.files.index(state['file_name'])
        else:
            self.current_shard_idx = min(state['current_shard_idx'], len(self.files) - 1)
        self.position = state['position']
        self.shard_step = int(state.get('shard_step', 0))
        self.load_shard()

    def _pick_next_shard(self):
        """Advance current_shard_idx, optionally weighted by self.mix."""
        if not self.mix:
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
            return

        # Bucket files by prefix.
        buckets = {}
        for f in self.files:
            base = os.path.basename(f)
            for pref in self.mix.keys():
                if base.startswith(pref):
                    buckets.setdefault(pref, []).append(f)
                    break
        if not buckets:
            self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
            return

        prefs = sorted(buckets.keys())
        weights = np.array([self.mix[p] for p in prefs], dtype=np.float64)
        weights = weights / weights.sum()

        rng = np.random.default_rng(self.shard_step)
        choice = int(rng.choice(len(prefs), p=weights))
        bucket = buckets[prefs[choice]]
        # Within the bucket, walk shards in order, indexed by how many times
        # we've drawn this bucket so far.
        draws_in_bucket = 0
        for s in range(self.shard_step + 1):
            r = np.random.default_rng(s)
            if int(r.choice(len(prefs), p=weights)) == choice:
                draws_in_bucket += 1
        idx_in_bucket = (draws_in_bucket - 1) % len(bucket)
        self.current_shard_idx = self.files.index(bucket[idx_in_bucket])

    def get_batch_cpu(self):
        """Return pinned CPU int64 tensors (x, y) sized (B, T)."""
        B, T = self.batch_size, self.seq_len
        tokens_needed = (B * T) + 1

        if self.position + tokens_needed > len(self.mmap):
            self._refresh_files()
            self.shard_step += 1
            self._pick_next_shard()
            self.position = self.rank * B * T
            self.load_shard()

        chunk = np.asarray(self.mmap[self.position : self.position + tokens_needed],
                           dtype=np.int64)
        x = torch.from_numpy(chunk[:-1]).view(B, T).pin_memory()
        y = torch.from_numpy(chunk[1:]).view(B, T).pin_memory()

        self.position += (B * T) * self.world_size
        return x, y


class CudaPrefetcher:
    """Double-buffered async H2D copy on a side stream."""

    def __init__(self, loader: DistributedShardLoader, device: str):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self._next_x = None
        self._next_y = None
        self._preload()

    def _preload(self):
        x_cpu, y_cpu = self.loader.get_batch_cpu()
        with torch.cuda.stream(self.stream):
            self._next_x = x_cpu.to(self.device, non_blocking=True)
            self._next_y = y_cpu.to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        x, y = self._next_x, self._next_y
        # Record current stream usage so the side stream doesn't free the
        # source pinned tensor before the compute kernels consume it.
        x.record_stream(torch.cuda.current_stream(self.device))
        y.record_stream(torch.cuda.current_stream(self.device))
        self._preload()
        return x, y

# =============================================================================
# 5. TRAINING UTILS
# =============================================================================

def get_lr(step: int, *, start_step: int, max_steps: int,
           schedule: str = SCHEDULE,
           resume_lr: float = RESUME_LR,
           peak_lr: float = PEAK_LR,
           floor_lr: float = FLOOR_LR,
           rewarmup_steps: int = REWARMUP_STEPS,
           cooldown_frac: float = COOLDOWN_FRAC) -> float:
    """LR for ``step``. Supports:
      - 'legacy_cosine': original cosine, global warmup from 0 to MAX_LR over WARMUP_STEPS.
      - 'cosine':       linear re-warmup from resume_lr->peak_lr over rewarmup_steps,
                        then cosine decay peak_lr->floor_lr over the remainder.
      - 'wsd':          linear re-warmup, then stable at peak, then linear cooldown
                        peak_lr->floor_lr over the last cooldown_frac of the run.
    All schedules treat ``start_step`` as the resume anchor.
    """
    if schedule == "legacy_cosine":
        if step < WARMUP_STEPS:
            return MAX_LR * (step / WARMUP_STEPS)
        if step > max_steps:
            return MIN_LR
        ratio = (step - WARMUP_STEPS) / max(1, (max_steps - WARMUP_STEPS))
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return MIN_LR + coeff * (MAX_LR - MIN_LR)

    # Re-warmup ramp (shared by wsd + cosine)
    rewarm_end = start_step + rewarmup_steps
    if step < rewarm_end:
        frac = (step - start_step) / max(1, rewarmup_steps)
        frac = max(0.0, min(1.0, frac))
        return resume_lr + frac * (peak_lr - resume_lr)

    if schedule == "cosine":
        decay_start = rewarm_end
        if step >= max_steps:
            return floor_lr
        ratio = (step - decay_start) / max(1, (max_steps - decay_start))
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return floor_lr + coeff * (peak_lr - floor_lr)

    # Default: WSD
    run_steps = max(1, max_steps - start_step)
    cooldown_len = max(1, int(run_steps * cooldown_frac))
    cooldown_start = max_steps - cooldown_len
    if step < cooldown_start:
        return peak_lr
    frac = (step - cooldown_start) / max(1, max_steps - cooldown_start)
    frac = max(0.0, min(1.0, frac))
    return peak_lr + frac * (floor_lr - peak_lr)

@torch.no_grad()
def estimate_loss(model, val_loader, eval_steps=20):
    model.eval()
    val_loader.current_shard_idx = 0
    val_loader.position = val_loader.rank * val_loader.batch_size * val_loader.seq_len
    val_loader.load_shard()
    prefetcher = CudaPrefetcher(val_loader, device)

    losses = torch.zeros(eval_steps, device=device)
    for k in range(eval_steps):
        x, y = prefetcher.next()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[k] = loss.detach()

    mean_loss = losses.mean()
    if ddp:
        torch.distributed.all_reduce(mean_loss, op=torch.distributed.ReduceOp.AVG)

    model.train()
    return mean_loss.item()


def run_inline_eval(raw_compiled_model, *, tag: str, tasks: str, limit,
                    global_step: int, wandb_run):
    """Run lm-eval-harness on rank 0 only against the live (in-memory) model.

    We unwrap torch.compile via ``_orig_mod`` to avoid recompilation churn from
    the variable shapes that lm-eval generates. Other ranks block on a barrier
    so DDP gradient buffers stay consistent for the next step.
    """
    if not EVAL_ENABLED:
        return
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    if not task_list:
        return

    if master_process:
        # Prefer the un-compiled module to skip recompilation during eval.
        eval_module = getattr(raw_compiled_model, "_orig_mod", raw_compiled_model)
        was_training = eval_module.training
        eval_module.eval()
        try:
            from eval_ivllm import IvLLMWrapper  # local import: keep lm-eval lazy
            from lm_eval import evaluator

            lm = IvLLMWrapper.from_model(
                eval_module, device=device, batch_size=EVAL_BATCH_SIZE,
                max_length=SEQ_LENGTH, dtype="bfloat16",
            )
            t_eval0 = time.time()
            res = evaluator.simple_evaluate(
                model=lm, tasks=task_list, limit=limit,
                batch_size=EVAL_BATCH_SIZE, num_fewshot=0,
            )
            t_eval = time.time() - t_eval0
        finally:
            if was_training:
                eval_module.train()

        flat = {}
        print(f"--- Inline eval [{tag}] @ step {global_step}  ({t_eval:.1f}s) ---")
        for t, metrics in res["results"].items():
            scalar = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
            print(f"  {t:25s}  " + "  ".join(f"{k}={v:.4f}" for k, v in scalar.items()))
            for k, v in scalar.items():
                flat[f"eval_{tag}/{t}/{k}"] = v
        flat[f"eval_{tag}/wall_sec"] = t_eval
        if wandb_run is not None:
            wandb_run.log(flat, step=global_step)

    if ddp:
        torch.distributed.barrier()

# =============================================================================
# 6. THE MAIN TRAINING LOOP
# =============================================================================

def main():
    train_loader = DistributedShardLoader(
        DATA_DIR, TRAIN_PREFIXES, MICRO_BATCH_SIZE, SEQ_LENGTH,
        ddp_rank, ddp_world_size, is_train=True, mix=DATA_MIX,
    )
    val_loader = DistributedShardLoader(
        DATA_DIR, (VAL_PREFIX,), MICRO_BATCH_SIZE, SEQ_LENGTH,
        ddp_rank, ddp_world_size, is_train=False,
    )

    if master_process:
        print(f"\n--- IvLLM Training Pipeline ---")
        print(f"GPUs: {ddp_world_size}  |  micro_bs={MICRO_BATCH_SIZE}  |  grad_accum={GRAD_ACCUM_STEPS}")
        print(f"Global Batch: {GLOBAL_BATCH_SIZE} seq ({TOKENS_PER_STEP:,} tok/step)")
        print(f"Data dir:     {DATA_DIR}  |  mix: {DATA_MIX or 'sequential'}")
        print(f"Schedule:     {SCHEDULE}  peak={PEAK_LR:g} floor={FLOOR_LR:g} "
              f"rewarmup={REWARMUP_STEPS} cooldown_frac={COOLDOWN_FRAC}")
        print(f"Train shards: {len(train_loader.files)} ({train_loader.total_tokens:,} tokens)")
        print(f"Val shards:   {len(val_loader.files)} ({val_loader.total_tokens:,} tokens)")

    raw_model = IvLLM().to(device)
    raw_model = torch.compile(raw_model, dynamic=False)
    if ddp:
        model = DDP(raw_model, device_ids=[ddp_local_rank],
                    gradient_as_bucket_view=True,
                    broadcast_buffers=False)
    else:
        model = raw_model

    param_dict = {pn: p for pn, p in raw_model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'embeddings' not in n]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'embeddings' in n]

    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8,
        fused=torch.cuda.is_available(),
    )

    steps_this_chunk = max(1, (NUM_EPOCHS * train_loader.total_tokens) // TOKENS_PER_STEP)
    # --- STATE RESUME LOGIC ---
    global_step = 0
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "ivllm_latest.pt")

    if os.path.exists(latest_ckpt):
        checkpoint = torch.load(latest_ckpt, map_location=device, weights_only=False)
        raw_model.load_state_dict(checkpoint['model_state_dict'])
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except (ValueError, KeyError) as e:
            if master_process:
                print(f"Optimizer state could not be restored ({e}); starting optimizer fresh.")
        global_step = int(checkpoint.get('global_step', 0))
        if 'dataloader_state' in checkpoint:
            try:
                train_loader.load_state_dict(checkpoint['dataloader_state'])
            except Exception as e:
                if master_process:
                    print(f"Dataloader state ignored ({e}); starting from shard 0.")
        if master_process:
            print(f"Resuming from checkpoint at Global Step {global_step}.")
    else:
        if master_process:
            print("No checkpoint found. Starting from scratch.")

    # Horizon: explicit step override > token-budget override > shard-derived fallback.
    if _ENV_MAX_STEPS is not None:
        max_steps = int(_ENV_MAX_STEPS)
    elif _ENV_TARGET_TOKENS is not None:
        max_steps = max(1, _parse_token_count(_ENV_TARGET_TOKENS) // TOKENS_PER_STEP)
    else:
        max_steps = global_step + steps_this_chunk
    start_step = global_step  # anchor for re-warmup
    if master_process:
        print(f"Target Horizon: Step {max_steps}  (start_step={start_step})")

    # --- wandb init (master only) ---
    wandb_run = None
    if master_process and WANDB_ENABLED:
        import wandb
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            resume="allow",
            config={
                "vocab_size": VOCAB_SIZE,
                "dim": DIM,
                "num_blocks": NUM_BLOCKS,
                "num_q_heads": NUM_Q_HEADS,
                "num_kv_groups": NUM_KV_GROUPS,
                "head_dim": HEAD_DIM,
                "seq_length": SEQ_LENGTH,
                "micro_batch_size": MICRO_BATCH_SIZE,
                "global_batch_size": GLOBAL_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "tokens_per_step": TOKENS_PER_STEP,
                "max_lr": MAX_LR,
                "min_lr": MIN_LR,
                "warmup_steps": WARMUP_STEPS,
                "schedule": SCHEDULE,
                "peak_lr": PEAK_LR,
                "floor_lr": FLOOR_LR,
                "resume_lr": RESUME_LR,
                "rewarmup_steps": REWARMUP_STEPS,
                "cooldown_frac": COOLDOWN_FRAC,
                "data_mix": DATA_MIX,
                "weight_decay": WEIGHT_DECAY,
                "num_epochs": NUM_EPOCHS,
                "checkpoint_interval": CHECKPOINT_INTERVAL,
                "data_dir": DATA_DIR,
                "train_shards": len(train_loader.files),
                "train_tokens": train_loader.total_tokens,
                "val_shards": len(val_loader.files),
                "val_tokens": val_loader.total_tokens,
                "world_size": ddp_world_size,
                "resume_step": global_step,
                "start_step": start_step,
                "max_steps": max_steps,
                "torch_version": torch.__version__,
            },
        )

    # Build prefetcher AFTER resume so it observes the resumed dataloader state.
    prefetcher = CudaPrefetcher(train_loader, device)

    model.train()
    t0 = time.time()
    total_tokens_window = 0

    # Inline-eval bookkeeping: anchor full-eval cadence to the resume token count
    # so we don't trigger immediately upon restart.
    full_interval_tokens = _parse_token_count(EVAL_FULL_INTERVAL_TOKENS)
    tokens_at_last_full_eval = global_step * TOKENS_PER_STEP
    last_inline_eval_step = global_step

    while global_step < max_steps:
        lr = get_lr(global_step, start_step=start_step, max_steps=max_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        # Accumulate loss as a GPU tensor — avoid .item() per microstep.
        loss_accum = torch.zeros((), device=device)

        for micro_step in range(GRAD_ACCUM_STEPS):
            x, y = prefetcher.next()
            sync_context = (
                model.no_sync()
                if (ddp and micro_step < GRAD_ACCUM_STEPS - 1)
                else nullcontext()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    loss = loss / GRAD_ACCUM_STEPS
                loss.backward()
                loss_accum += loss.detach()

            total_tokens_window += (MICRO_BATCH_SIZE * SEQ_LENGTH * ddp_world_size)

        if ddp:
            torch.distributed.all_reduce(loss_accum, op=torch.distributed.ReduceOp.AVG)

        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
        optimizer.step()

        global_step += 1

        if master_process and global_step % LOG_INTERVAL == 0:
            # Single sync point per logging window.
            loss_val = loss_accum.item()
            gnorm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = total_tokens_window / dt
            step_ms = (dt / LOG_INTERVAL) * 1000.0
            print(f"Step: {global_step:6d}/{max_steps} | Loss: {loss_val:.4f} | "
                  f"LR: {lr:.2e} | gnorm: {gnorm_val:.3f} | "
                  f"Speed: {tokens_per_sec:,.0f} tok/s | {step_ms:.0f} ms/step")
            if wandb_run is not None:
                wandb_run.log({
                    "train/loss": loss_val,
                    "train/lr": lr,
                    "train/grad_norm": gnorm_val,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/step_ms": step_ms,
                    "train/tokens_seen": global_step * TOKENS_PER_STEP,
                    "train/epoch_frac": global_step / max(1, max_steps),
                }, step=global_step)
            t0 = time.time()
            total_tokens_window = 0

        if global_step % CHECKPOINT_INTERVAL == 0:
            val_loss = estimate_loss(model, val_loader)
            if master_process:
                print(f"--- Checkpoint (Step {global_step} | Val Loss: {val_loss:.4f}) ---")
                if wandb_run is not None:
                    wandb_run.log({"val/loss": val_loss}, step=global_step)

                prev_ckpt = os.path.join(CHECKPOINT_DIR, "ivllm_prev.pt")
                if os.path.exists(latest_ckpt):
                    os.replace(latest_ckpt, prev_ckpt)

                tmp_ckpt = latest_ckpt + ".tmp"
                torch.save({
                    'global_step': global_step,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataloader_state': train_loader.state_dict(),
                }, tmp_ckpt)
                os.replace(tmp_ckpt, latest_ckpt)
            if ddp:
                torch.distributed.barrier()
            # Rebuild the prefetcher because eval rewound the val loader.
            prefetcher = CudaPrefetcher(train_loader, device)

        # --- Inline lm-eval-harness hooks ---
        if EVAL_ENABLED and global_step > last_inline_eval_step:
            tokens_seen = global_step * TOKENS_PER_STEP
            do_full = (full_interval_tokens > 0 and
                       tokens_seen - tokens_at_last_full_eval >= full_interval_tokens)
            do_quick = (EVAL_INTERVAL_STEPS > 0 and
                        global_step % EVAL_INTERVAL_STEPS == 0)
            if do_full:
                run_inline_eval(raw_model, tag="full", tasks=EVAL_FULL_TASKS,
                                limit=None, global_step=global_step,
                                wandb_run=wandb_run)
                tokens_at_last_full_eval = tokens_seen
                last_inline_eval_step = global_step
                # Eval drove varied shapes through the model; refresh prefetcher
                # so the next training step starts from a clean side stream.
                prefetcher = CudaPrefetcher(train_loader, device)
            elif do_quick:
                run_inline_eval(raw_model, tag="quick", tasks=EVAL_QUICK_TASKS,
                                limit=EVAL_QUICK_LIMIT, global_step=global_step,
                                wandb_run=wandb_run)
                last_inline_eval_step = global_step
                prefetcher = CudaPrefetcher(train_loader, device)

    val_loss = estimate_loss(model, val_loader)
    if master_process:
        print("\n===========================================")
        print(f"SUCCESS: Reached horizon at Global Step {global_step}.")
        print(f"Final Val Loss: {val_loss:.4f}")
        if wandb_run is not None:
            wandb_run.log({"val/loss": val_loss}, step=global_step)
        tmp_ckpt = latest_ckpt + ".tmp"
        torch.save({
            'global_step': global_step,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataloader_state': train_loader.state_dict(),
        }, tmp_ckpt)
        os.replace(tmp_ckpt, latest_ckpt)
        print("===========================================\n")
        if wandb_run is not None:
            wandb_run.finish()

    if ddp:
        torch.distributed.barrier()
        destroy_process_group()


if __name__ == "__main__":
    main()
