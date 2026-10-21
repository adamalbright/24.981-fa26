#!/usr/bin/env python3
"""Write the final tab-delimited column of a file as a one-column text file.

Usage:
    python scripts/extract_last_column.py \
        --input testing/AlbrightHayes2003.arpabet.txt \
        --output testing/AlbrightHayes2003.arpabet.test.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.input.open(encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source, delimiter="\t"))
    if any(not row for row in rows):
        raise ValueError("Input contains an empty row.")

    with args.output.open("w", encoding="utf-8", newline="") as destination:
        destination.write("\n".join(row[-1] for row in rows[1:]) + "\n")

    print(f"Wrote {len(rows) - 1} data rows to {args.output}")


if __name__ == "__main__":
    main()
