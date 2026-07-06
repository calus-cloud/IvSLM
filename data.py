"""Data utilities for IvLLM.

Subcommands
-----------
  download-fineweb [N]              Pull the first N fineweb100B-gpt2 shards (default 1030).
  download-math   [--tokens N]      Stream open-web-math, tokenize + shard into bins.
  download-code   [--tokens N]      Stream HuggingFaceTB/smollm-corpus python-edu,
                                    tokenize + shard into bins.
  encode-jsonl <path> [--field text] [--prefix custom_train]
                                    Tokenize a .jsonl(.gz/.zst) file or directory.
  encode-text <path> [--prefix custom_train]
                                    Tokenize plain .txt files (each file = one doc).

All bin shards are written into the same DATA_DIR (default ``fineweb100B``) in the
nanoGPT/FineWeb format (256 int32 header + uint16 tokens) so the IvLLM dataloader
picks them up automatically.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import io
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download

# ------- Shard format constants (must match ivllm.py) -------
HEADER_INTS = 256
MAGIC_NUMBER = 20240520
VERSION = 1
EOT_TOKEN = 50256  # GPT-2 <|endoftext|>
SHARD_TOKENS = 100_000_000  # 100M tokens per shard, like fineweb

DATA_DIR = os.environ.get("IVLLM_DATA_DIR", "fineweb100B")


# =============================================================================
# Shard I/O
# =============================================================================

def write_shard(path: str, tokens: np.ndarray) -> None:
    assert tokens.dtype == np.uint16
    header = np.zeros(HEADER_INTS, dtype=np.int32)
    header[0] = MAGIC_NUMBER
    header[1] = VERSION
    header[2] = len(tokens)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())
    os.replace(tmp, path)


def next_shard_index(out_dir: str, prefix: str) -> int:
    existing = sorted(glob.glob(os.path.join(out_dir, f"{prefix}_*.bin")))
    if not existing:
        return 0
    last = os.path.basename(existing[-1])
    # filename pattern <prefix>_NNNNNN.bin
    try:
        return int(last[len(prefix) + 1 : -4]) + 1
    except ValueError:
        return len(existing)


# =============================================================================
# Multiprocess tiktoken pipeline
# =============================================================================

_ENC: Optional[tiktoken.Encoding] = None


def _worker_init() -> None:
    global _ENC
    _ENC = tiktoken.get_encoding("gpt2")


def _worker_encode(text: str) -> np.ndarray:
    """Encode one document; prepend EOT as a doc boundary."""
    ids = _ENC.encode_ordinary(text)
    out = np.empty(len(ids) + 1, dtype=np.uint16)
    out[0] = EOT_TOKEN
    out[1:] = ids
    return out


def encode_and_shard(
    docs: Iterable[str],
    out_dir: str,
    prefix: str,
    target_tokens: Optional[int] = None,
    shard_tokens: int = SHARD_TOKENS,
    num_procs: Optional[int] = None,
    chunksize: int = 16,
) -> int:
    """Tokenize ``docs`` and write fixed-size .bin shards.

    Returns the total number of tokens written.
    """
    os.makedirs(out_dir, exist_ok=True)
    nprocs = num_procs or max(1, (os.cpu_count() or 4) - 2)
    shard_idx = next_shard_index(out_dir, prefix)
    print(f"[encode] prefix={prefix} dir={out_dir} workers={nprocs} "
          f"shard_tokens={shard_tokens:,} starting_at_shard={shard_idx}")

    buf = np.empty(shard_tokens, dtype=np.uint16)
    buf_pos = 0
    total = 0

    with mp.Pool(nprocs, initializer=_worker_init) as pool:
        for arr in pool.imap_unordered(_worker_encode, docs, chunksize=chunksize):
            n = len(arr)
            # If this doc would overflow the current shard, fill what we can
            # then flush, and start the next shard with the remainder.
            offset = 0
            while n > 0:
                space = shard_tokens - buf_pos
                take = min(space, n)
                buf[buf_pos : buf_pos + take] = arr[offset : offset + take]
                buf_pos += take
                offset += take
                n -= take
                total += take

                if buf_pos == shard_tokens:
                    shard_path = os.path.join(out_dir, f"{prefix}_{shard_idx:06d}.bin")
                    write_shard(shard_path, buf)
                    print(f"[encode]   wrote {shard_path}  ({total:,} tokens cumulative)")
                    shard_idx += 1
                    buf_pos = 0

                    if target_tokens is not None and total >= target_tokens:
                        pool.terminate()
                        return total

    # Flush trailing partial shard so no tokens are lost.
    if buf_pos > 0:
        shard_path = os.path.join(out_dir, f"{prefix}_{shard_idx:06d}.bin")
        write_shard(shard_path, buf[:buf_pos].copy())
        print(f"[encode]   wrote {shard_path}  (partial: {buf_pos:,} tokens)")
        total_partial = buf_pos
    else:
        total_partial = 0

    print(f"[encode] DONE prefix={prefix}: {total:,} tokens "
          f"(+{total_partial:,} in trailing shard)")
    return total


# =============================================================================
# Streaming sources
# =============================================================================

def _stream_hf(repo: str, *, config: Optional[str] = None, split: str = "train",
               text_field: str = "text") -> Iterator[str]:
    """Yield text fields from a streaming HuggingFace dataset."""
    from datasets import load_dataset  # local import to keep base import light

    ds = load_dataset(repo, name=config, split=split, streaming=True)
    for row in ds:
        text = row.get(text_field)
        if text:
            yield text


def stream_math() -> Iterator[str]:
    """open-web-math: ~14B GPT-2 tokens of math web pages."""
    return _stream_hf("open-web-math/open-web-math", text_field="text")


def stream_code() -> Iterator[str]:
    """HuggingFaceTB/smollm-corpus python-edu: ~4B tokens of educational Python."""
    return _stream_hf(
        "HuggingFaceTB/smollm-corpus", config="python-edu", text_field="text"
    )


# =============================================================================
# File-based sources for arbitrary user data
# =============================================================================

def _open_maybe_compressed(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.endswith(".zst"):
        import zstandard as zstd  # optional dep; only needed for .zst
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _expand_paths(path: str) -> list[str]:
    p = Path(path)
    if p.is_dir():
        return sorted(str(x) for x in p.rglob("*") if x.is_file())
    return [str(p)]


def iter_jsonl(path: str, field: str = "text") -> Iterator[str]:
    for fp in _expand_paths(path):
        with _open_maybe_compressed(fp) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get(field)
                if text:
                    yield text


def iter_text_files(path: str) -> Iterator[str]:
    """Each file is treated as one document."""
    for fp in _expand_paths(path):
        with _open_maybe_compressed(fp) as fh:
            doc = fh.read()
            if doc:
                yield doc


# =============================================================================
# Fineweb pre-tokenized shard download (original behavior, kept)
# =============================================================================

def download_fineweb(num_chunks: int = 1030, out_dir: str = DATA_DIR) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _get(fname: str) -> None:
        if not os.path.exists(os.path.join(out_dir, fname)):
            hf_hub_download(
                repo_id="kjj0/fineweb100B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=out_dir,
            )

    _get("fineweb_val_%06d.bin" % 0)
    for i in range(1, num_chunks + 1):
        _get("fineweb_train_%06d.bin" % i)


# =============================================================================
# CLI
# =============================================================================

def _parse_tokens(s: str) -> int:
    """Accept things like '1_000_000_000', '1e9', '1B', '500M'."""
    s = s.strip().replace("_", "")
    mult = 1
    if s[-1] in "Kk":
        mult, s = 1_000, s[:-1]
    elif s[-1] in "Mm":
        mult, s = 1_000_000, s[:-1]
    elif s[-1] in "Bb":
        mult, s = 1_000_000_000, s[:-1]
    return int(float(s) * mult)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=DATA_DIR,
                   help=f"Directory for .bin shards (default: {DATA_DIR})")
    p.add_argument("--workers", type=int, default=None,
                   help="Tokenizer worker count (default: cpu_count - 2)")
    sub = p.add_subparsers(dest="cmd", required=True)

    fw = sub.add_parser("download-fineweb", help="Pull fineweb100B-gpt2 shards.")
    fw.add_argument("num_chunks", nargs="?", type=int, default=1030)

    m = sub.add_parser("download-math", help="Stream + tokenize open-web-math.")
    m.add_argument("--tokens", type=_parse_tokens, default=_parse_tokens("1B"))
    m.add_argument("--prefix", default="math_train")

    c = sub.add_parser("download-code", help="Stream + tokenize python-edu.")
    c.add_argument("--tokens", type=_parse_tokens, default=_parse_tokens("1B"))
    c.add_argument("--prefix", default="code_train")

    ej = sub.add_parser("encode-jsonl", help="Tokenize a jsonl file or directory.")
    ej.add_argument("path")
    ej.add_argument("--field", default="text")
    ej.add_argument("--prefix", required=True)
    ej.add_argument("--tokens", type=_parse_tokens, default=None,
                    help="Optional cap on total tokens.")

    et = sub.add_parser("encode-text", help="Tokenize plain text file(s).")
    et.add_argument("path")
    et.add_argument("--prefix", required=True)
    et.add_argument("--tokens", type=_parse_tokens, default=None)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.cmd == "download-fineweb":
        download_fineweb(args.num_chunks, args.out_dir)
        return 0

    if args.cmd == "download-math":
        encode_and_shard(stream_math(), args.out_dir, args.prefix,
                         target_tokens=args.tokens, num_procs=args.workers)
        return 0

    if args.cmd == "download-code":
        encode_and_shard(stream_code(), args.out_dir, args.prefix,
                         target_tokens=args.tokens, num_procs=args.workers)
        return 0

    if args.cmd == "encode-jsonl":
        encode_and_shard(iter_jsonl(args.path, args.field), args.out_dir,
                         args.prefix, target_tokens=args.tokens,
                         num_procs=args.workers)
        return 0

    if args.cmd == "encode-text":
        encode_and_shard(iter_text_files(args.path), args.out_dir,
                         args.prefix, target_tokens=args.tokens,
                         num_procs=args.workers)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
  
