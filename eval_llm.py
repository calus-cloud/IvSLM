"""IvLLM evaluator built on top of `lm-eval` harness.

Tokenizer: tiktoken gpt2 (matching pretraining).
Loads a training checkpoint, wraps the model as a `lm_eval.api.model.LM`, and
runs the requested suite of tasks.

Usage (run after training is paused or on idle GPUs):

    python eval_ivllm.py \\
        --ckpt checkpoints/ivllm_latest.pt \\
        --tasks hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai \\
        --batch-size 16 --device cuda:0 --output results.json

Only loglikelihood-style tasks are implemented (covers all standard small-model
zero-shot benchmarks). `generate_until` raises NotImplementedError.
"""

import argparse
import json
import math
import os
import sys
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F
import tiktoken
from tqdm import tqdm

from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# NOTE: Do NOT `import ivllm` at module scope. When eval is invoked from inside
# a running torchrun process, the live script is loaded as ``__main__``; a
# top-level ``import ivllm`` would re-execute its module body (including the
# NCCL ``init_process_group``) and crash the training run.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------
@register_model("ivllm")
class IvLLMWrapper(LM):
    def __init__(
        self,
        ckpt: str | None = None,
        device: str = "cuda:0",
        batch_size: int = 16,
        max_length: int = 2048,
        dtype: str = "bfloat16",
        model: torch.nn.Module | None = None,
    ):
        super().__init__()
        self._device = torch.device(device)
        self.batch_size_per_gpu = int(batch_size)
        self._max_length = int(max_length)
        self._dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                       "float32": torch.float32}[dtype]

        # tokenizer
        self.tok = tiktoken.get_encoding("gpt2")
        self._eot = self.tok.eot_token  # 50256

        if model is not None:
            # In-training mode: use the live model as-is (don't move/recast it).
            self.model = model
            self._owns_model = False
        else:
            assert ckpt is not None, "Provide either `ckpt` or `model`"
            print(f"[ivllm] loading {ckpt} ...")
            # Lazy import — only safe outside of an active torchrun process.
            import ivllm as _ivllm
            m = _ivllm.IvLLM()
            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
            # Strip "_orig_mod." prefix if checkpoint was saved from torch.compile().
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            missing, unexpected = m.load_state_dict(sd, strict=False)
            if missing:
                print(f"[ivllm] missing keys: {len(missing)} (first 5: {missing[:5]})")
            if unexpected:
                print(f"[ivllm] unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
            m.to(self._device).to(self._dtype).eval()
            self.model = m
            self._owns_model = True

        # Misc bookkeeping
        self._rank = 0
        self._world_size = 1

    @classmethod
    def from_model(cls, model: torch.nn.Module, *, device, batch_size: int = 16,
                   max_length: int = 2048, dtype: str = "bfloat16"):
        return cls(ckpt=None, device=str(device), batch_size=batch_size,
                   max_length=max_length, dtype=dtype, model=model)

    # ---- harness boilerplate ----
    @property
    def eot_token_id(self):
        return self._eot

    @property
    def device(self):
        return self._device

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, **_) -> List[int]:
        return self.tok.encode_ordinary(string)

    def tok_decode(self, tokens) -> str:
        if torch.is_tensor(tokens):
            tokens = tokens.tolist()
        return self.tok.decode(tokens)

    # ---- core scoring ----
    @torch.no_grad()
    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B, T) int64 on device → logits (B, T, V) in self._dtype."""
        with torch.autocast(device_type=self.device.type, dtype=self._dtype):
            logits, _ = self.model(input_ids)
        return logits

    def _encode_pair(self, context: str, continuation: str) -> Tuple[List[int], List[int]]:
        if context == "":
            # Bare continuation: prefix with EOT so the model has something to
            # condition on, then score the continuation.
            ctx_ids = [self._eot]
        else:
            ctx_ids = self.tok.encode_ordinary(context)
        cont_ids = self.tok.encode_ordinary(continuation)
        return ctx_ids, cont_ids

    @torch.no_grad()
    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        """Score (context, continuation) pairs. Returns [(logprob, is_greedy), ...]."""
        # Build the full input for every request and remember the continuation slice.
        scored = []
        encoded: List[Tuple[List[int], int]] = []  # (full_input_ids, continuation_len)
        for req in requests:
            ctx, cont = req.args
            ctx_ids, cont_ids = self._encode_pair(ctx, cont)
            # Truncate from the left if too long, keeping continuation intact.
            full = ctx_ids + cont_ids
            if len(full) > self.max_length:
                overflow = len(full) - self.max_length
                full = full[overflow:]
            # Need at least one context token for prediction of cont[0].
            cont_len = min(len(cont_ids), len(full) - 1)
            encoded.append((full, cont_len))

        # Sort by length descending for efficient batched padding.
        order = sorted(range(len(encoded)), key=lambda i: -len(encoded[i][0]))
        results: List[Tuple[float, bool]] = [None] * len(encoded)  # type: ignore

        pbar = tqdm(total=len(encoded), desc="loglik", leave=False)
        for batch_start in range(0, len(order), self.batch_size_per_gpu):
            idxs = order[batch_start: batch_start + self.batch_size_per_gpu]
            seqs = [encoded[i][0] for i in idxs]
            cont_lens = [encoded[i][1] for i in idxs]
            max_len = max(len(s) for s in seqs)

            # Left-pad with EOT so the predicted positions line up at the right.
            padded = torch.full((len(seqs), max_len), self._eot,
                                dtype=torch.long, device=self.device)
            for row, s in enumerate(seqs):
                padded[row, max_len - len(s):] = torch.tensor(s, device=self.device)

            logits = self._model_logits(padded).float()  # (B, T, V)

            # Score each continuation: target = padded[:, 1:], prediction logits = logits[:, :-1].
            for row, (full, cont_len) in enumerate(zip(seqs, cont_lens)):
                # Position of last full-input token within `padded`:
                end = max_len  # padded length
                # Continuation tokens occupy positions [end - cont_len, end) in `padded`.
                target_slice = padded[row, end - cont_len: end]
                # Predicted-from positions are [end - cont_len - 1, end - 1) in logits.
                pred_slice = logits[row, end - cont_len - 1: end - 1]
                log_probs = F.log_softmax(pred_slice, dim=-1)
                tok_lp = log_probs.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)
                total_lp = float(tok_lp.sum().item())
                greedy = torch.argmax(pred_slice, dim=-1)
                is_greedy = bool(torch.equal(greedy, target_slice))
                results[idxs[row]] = (total_lp, is_greedy)
            pbar.update(len(idxs))
        pbar.close()
        return results

    @torch.no_grad()
    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        """Rolling perplexity over a long string (used by wikitext/lambada-rolling)."""
        out: List[float] = []
        for req in tqdm(requests, desc="loglik_rolling", leave=False):
            (string,) = req.args
            ids = [self._eot] + self.tok.encode_ordinary(string)
            total_lp = 0.0
            stride = self.max_length
            for start in range(0, len(ids) - 1, stride - 1):
                window = ids[start: start + self.max_length]
                if len(window) < 2:
                    break
                inp = torch.tensor(window, device=self.device, dtype=torch.long)[None]
                logits = self._model_logits(inp).float()
                lp = F.log_softmax(logits[0, :-1], dim=-1)
                tgt = inp[0, 1:]
                total_lp += float(lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item())
            out.append(total_lp)
        return out

    def generate_until(self, requests):
        raise NotImplementedError(
            "generate_until is not implemented for IvLLM (loglikelihood-style "
            "tasks only). Use tasks like hellaswag/arc/piqa/winogrande/lambada."
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
DEFAULT_TASKS = (
    "hellaswag,arc_easy,arc_challenge,piqa,winogrande,lambada_openai"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ivllm_*.pt")
    ap.add_argument("--tasks", default=DEFAULT_TASKS,
                    help="comma-sep list of lm-eval tasks")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--limit", type=int, default=None,
                    help="optional cap on examples per task (smoke test)")
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--output", default=None,
                    help="path to write JSON results")
    ap.add_argument("--wandb", action="store_true",
                    help="log results to wandb (project=ivllm, group=eval)")
    args = ap.parse_args()

    lm = IvLLMWrapper(
        ckpt=args.ckpt,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        dtype=args.dtype,
    )

    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    print(f"[ivllm] running tasks: {task_list}")
    res = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
    )

    summary = {}
    for t, metrics in res["results"].items():
        summary[t] = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    print("\n=== RESULTS ===")
    for t, m in summary.items():
        print(f"{t:25s}  " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "results": summary,
                       "config": res.get("config", {})}, f, indent=2)
        print(f"[ivllm] wrote {args.output}")

    if args.wandb:
        import wandb
        wandb.init(project="ivllm", job_type="eval",
                   name=os.path.basename(args.ckpt) + "-eval",
                   config={"ckpt": args.ckpt, "tasks": task_list,
                           "num_fewshot": args.num_fewshot})
        flat = {}
        for t, m in summary.items():
            for k, v in m.items():
                flat[f"eval/{t}/{k}"] = v
        wandb.log(flat)
        wandb.finish()


if __name__ == "__main__":
    main()
