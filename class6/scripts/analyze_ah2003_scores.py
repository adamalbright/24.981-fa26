#!/usr/bin/env python3
"""Correlate phonotactic perplexity with Albright & Hayes acceptability ratings.

Example:
    python scripts/analyze_ah2003_scores.py \
        --data testing/AlbrightHayes2003.arpabet.txt \
        --scores testing/AlbrightHayes2003.arpabet.scores.tsv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def pearson(values_a: list[float], values_b: list[float]) -> float:
    """Return Pearson's r, with a clear error for an unvarying column."""
    if len(values_a) != len(values_b) or len(values_a) < 2:
        raise ValueError("Correlation requires at least two paired observations.")
    mean_a = sum(values_a) / len(values_a)
    mean_b = sum(values_b) / len(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denominator = math.sqrt(sum(value**2 for value in centered_a) * sum(value**2 for value in centered_b))
    if denominator == 0:
        raise ValueError("Correlation is undefined for a constant column.")
    return sum(a * b for a, b in zip(centered_a, centered_b)) / denominator


def average_ranks(values: list[float]) -> list[float]:
    """Return ascending ranks, assigning tied values their mean rank."""
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        rank = (start + 1 + end) / 2  # ranks are one-indexed
        for index, _ in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="AlbrightHayes2003.arpabet.txt")
    parser.add_argument("--scores", type=Path, required=True, help="Output from trained/score.py")
    args = parser.parse_args()

    with args.data.open(encoding="utf-8", newline="") as source:
        stimuli = list(csv.DictReader(source, delimiter="\t"))
    with args.scores.open(encoding="utf-8", newline="") as source:
        scores = list(csv.DictReader(source, delimiter="\t"))

    if len(stimuli) != len(scores):
        raise ValueError(f"Row-count mismatch: {len(stimuli)} stimuli but {len(scores)} scores.")
    for line_number, (stimulus, score) in enumerate(zip(stimuli, scores), start=2):
        if stimulus["ARPABET_1_BEFORE_VOWEL"] != score["transcription"]:
            raise ValueError(f"Transcription mismatch on line {line_number}; files are not aligned.")

    perplexity = [float(row["perplexity"]) for row in scores]
    print(f"Aligned {len(stimuli)} scored stimuli by their ARPABET transcription.")
    for column in ("AH2003", "Replicate2006"):
        paired = [
            (perplexity[index], float(stimulus[column]))
            for index, stimulus in enumerate(stimuli)
            if stimulus[column].strip()
        ]
        score_values, ratings = map(list, zip(*paired))
        spearman = pearson(average_ranks(score_values), average_ranks(ratings))
        print(
            f"{column}: n={len(paired)}, Pearson r={pearson(score_values, ratings):.4f}, "
            f"Spearman rho={spearman:.4f}"
        )


if __name__ == "__main__":
    main()
