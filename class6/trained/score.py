#!/usr/bin/env python3
"""Score ARPABET transcriptions with a phon_model.pt checkpoint.

The input must contain one whitespace-delimited token sequence per line, for
example ``B 1 AY Z``.  Scores include the probability of the final ``<eos>``
token.  Lower negative log probability (NLL), bits, and bits per token are
better; the latter is most appropriate for comparing words of different
lengths.

Example:
    python trained/score.py \
        --checkpoint trained/phon_model.pt \
        --input testing/AlbrightHayes2003.arpabet.test.txt \
        --output testing/AlbrightHayes2003.arpabet.scores.tsv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Transformer block; must match the architecture used for training."""

    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.h, self.p = heads, dropout
        self.ln1 = nn.LayerNorm(dimension)
        self.ln2 = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.proj = nn.Linear(dimension, dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 4 * dimension), nn.GELU(), nn.Linear(4 * dimension, dimension)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = inputs.shape
        query, key, value = self.qkv(self.ln1(inputs)).split(channels, dim=2)
        query, key, value = [
            tensor.view(batch_size, sequence_length, self.h, channels // self.h).transpose(1, 2)
            for tensor in (query, key, value)
        ]
        attended = F.scaled_dot_product_attention(
            query, key, value, is_causal=True,
            dropout_p=self.p if self.training else 0.0,
        )
        inputs = inputs + self.drop(self.proj(attended.transpose(1, 2).reshape(batch_size, sequence_length, channels)))
        return inputs + self.drop(self.mlp(self.ln2(inputs)))


class GPT(nn.Module):
    """Transformer language model; must match the architecture used for training."""

    def __init__(self, vocabulary_size: int, dimension: int, layers: int, heads: int, dropout: float, max_length: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocabulary_size, dimension)
        self.pos = nn.Embedding(max_length, dimension)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(Block(dimension, heads, dropout) for _ in range(layers))
        self.ln = nn.LayerNorm(dimension)
        self.head = nn.Linear(dimension, vocabulary_size, bias=False)
        self.head.weight = self.tok.weight

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.shape[1]
        hidden = self.drop(
            self.tok(inputs) + self.pos(torch.arange(sequence_length, device=inputs.device))
        )
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(self.ln(hidden))


def load_checkpoint(path: Path, device: str) -> dict:
    """Load checkpoints created by the course training script across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument
        return torch.load(path, map_location=device)


@torch.inference_mode()
def score(model: GPT, token_ids: list[int], device: str) -> tuple[float, int]:
    """Return sequence NLL in nats and the number of predicted tokens."""
    inputs = torch.tensor([token_ids[:-1]], device=device)
    targets = torch.tensor(token_ids[1:], device=device)
    log_probabilities = F.log_softmax(model(inputs)[0], dim=-1)
    nll = -log_probabilities.gather(1, targets[:, None]).sum().item()
    return nll, len(targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()

    device = "mps" if args.device == "auto" and torch.backends.mps.is_available() else "cpu"
    if args.device != "auto":
        device = args.device
    checkpoint = load_checkpoint(args.checkpoint, device)
    vocabulary, model_args = checkpoint["itos"], checkpoint["args"]
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    bos, eos = token_to_id["<bos>"], token_to_id["<eos>"]

    model = GPT(
        len(vocabulary), model_args["dim"], model_args["layers"], model_args["heads"],
        model_args["dropout"], model_args["max_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with args.input.open(encoding="utf-8") as source:
        sequences = [" ".join(line.split()) for line in source if line.strip()]

    rows = []
    for line_number, sequence in enumerate(sequences, start=1):
        tokens = sequence.split()
        unknown = sorted(set(tokens) - token_to_id.keys())
        if unknown:
            raise ValueError(f"Line {line_number} contains tokens absent from the checkpoint vocabulary: {unknown}")
        token_ids = [bos] + [token_to_id[token] for token in tokens] + [eos]
        if len(token_ids) > model_args["max_len"]:
            raise ValueError(f"Line {line_number} has {len(token_ids)} tokens including boundaries; checkpoint maximum is {model_args['max_len']}.")
        nll_nats, predicted_tokens = score(model, token_ids, device)
        nll_bits = nll_nats / math.log(2)
        bits_per_token = nll_bits / predicted_tokens
        rows.append([line_number, sequence, predicted_tokens, f"{nll_nats:.6f}", f"{nll_bits:.6f}", f"{bits_per_token:.6f}", f"{2 ** bits_per_token:.6f}"])

    with args.output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["line", "transcription", "predicted_tokens_including_eos", "nll_nats", "nll_bits", "bits_per_token", "perplexity"])
        writer.writerows(rows)
    print(f"Scored {len(rows)} sequences on {device}; wrote {args.output}")


if __name__ == "__main__":
    main()
