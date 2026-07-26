#!/usr/bin/env python3
"""Data prep: a small-vocab BPE over WikiText-103, packed into flat token arrays.

Vocab is deliberately small (4096). At V=50304 the tied embed/lm_head matmul is
~91% of FLOPs at d=128, so the transformer body — the only thing growth shrinks —
barely moves the compute budget. Body FLOPs dominate only when d > V/(12L); at
V=4096, L=6 that threshold is d>57, so the 128->256 growth range actually drives
compute and N(t) is measurable.
"""

import argparse
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
VOCAB_SIZE = 4096
# Enough for the largest control arm (2000 steps x 8 x 256 tokens) with no repeats.
N_TRAIN_DOCS = 300_000


def _tokenizer_path() -> Path:
    return DATA_DIR / f"bpe{VOCAB_SIZE}.json"


def prepare(seq_len: int = 256):
    """Train the BPE, tokenize, pack into flat uint16 arrays. Idempotent."""
    from datasets import load_dataset
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (DATA_DIR / "train.npy").exists() and (DATA_DIR / "val.npy").exists():
        print(f"Data already prepared in {DATA_DIR}")
        return

    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    train_docs = ds["train"].select(range(min(N_TRAIN_DOCS, len(ds["train"]))))["text"]
    val_docs = ds["validation"]["text"]

    tok_path = _tokenizer_path()
    if tok_path.exists():
        tok = Tokenizer.from_file(str(tok_path))
    else:
        print(f"Training {VOCAB_SIZE}-token BPE...")
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE, special_tokens=["<unk>", "<eos>"], show_progress=True
        )
        tok.train_from_iterator((t for t in train_docs if t.strip()), trainer=trainer)
        tok.save(str(tok_path))
    eos = tok.token_to_id("<eos>")

    def pack(docs, name):
        ids = []
        batch = [t for t in docs if t.strip()]
        for i in range(0, len(batch), 10_000):
            for enc in tok.encode_batch(batch[i : i + 10_000]):
                ids.extend(enc.ids)
                ids.append(eos)
            print(f"  {name}: {len(ids):,} tokens", end="\r")
        arr = np.array(ids, dtype=np.uint16)
        arr = arr[: (len(arr) // seq_len) * seq_len]
        np.save(DATA_DIR / f"{name}.npy", arr)
        print(f"  {name}: {len(arr):,} tokens -> {len(arr) // seq_len:,} seqs of {seq_len}")

    pack(train_docs, "train")
    pack(val_docs, "val")


class TokenLoader:
    """Deterministic batch sampler over a flat token array.

    Seeded independently of model init so every arm sees the same data order —
    the control compares trajectories, not data luck.
    """

    def __init__(self, split: str, batch_size: int, seq_len: int, seed: int = 0):
        self.tokens = np.load(DATA_DIR / f"{split}.npy")
        self.n_seqs = len(self.tokens) // seq_len
        self.batch_size, self.seq_len = batch_size, seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self):
        import torch

        idx = self.rng.integers(0, self.n_seqs, size=self.batch_size)
        starts = idx * self.seq_len
        x = np.stack([self.tokens[s : s + self.seq_len] for s in starts]).astype(np.int64)
        return torch.from_numpy(x)

    def eval_batches(self, n: int):
        """Fixed, non-random slice — identical across arms and seeds."""
        import torch

        for i in range(n):
            s = i * self.batch_size
            if s + self.batch_size > self.n_seqs:
                break
            starts = np.arange(s, s + self.batch_size) * self.seq_len
            x = np.stack([self.tokens[t : t + self.seq_len] for t in starts]).astype(np.int64)
            yield torch.from_numpy(x)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=256)
    prepare(p.parse_args().seq_len)
