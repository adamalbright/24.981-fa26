#!/usr/bin/env python3
"""Append ARPABET transcriptions to the Albright & Hayes (2003) stimuli.

The source file is tab-delimited and its second column contains Unicode IPA.
Two columns are appended:

* ``ARPABET``: phones separated by one space.
* ``ARPABET_1_BEFORE_VOWEL``: the same pronunciation, with a standalone ``1``
  token immediately before every vowel phone.

Usage:
    python scripts/convert_ah2003_to_arpabet.py \
        --input testing/AlbrightHayes2003.txt \
        --output testing/AlbrightHayes2003.arpabet.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# Longest IPA sequences must precede their component symbols.  The mappings
# follow the phonemic values used in this class's source transcription (e.g.,
# /e/ -> EY, /u/ -> UW, and /ɛə/ -> EH R).
IPA_TO_ARPABET: dict[str, tuple[str, ...]] = {
    "tʃ": ("CH",),
    "dʒ": ("JH",),
    "aɪ": ("AY",),
    "aʊ": ("AW",),
    "ɔɪ": ("OY",),
    "oʊ": ("OW",),
    "ɛə": ("EH", "R"),
    "ɪə": ("IH", "R"),
    "ʊə": ("UH", "R"),
    "ər": ("ER",),
    "i": ("IY",),
    "ɪ": ("IH",),
    "e": ("EY",),
    "ɛ": ("EH",),
    "æ": ("AE",),
    "ɑ": ("AA",),
    "ɔ": ("AO",),
    "o": ("OW",),
    "u": ("UW",),
    "ʊ": ("UH",),
    "ʌ": ("AH",),
    "ə": ("AH",),
    "ɜ": ("ER",),
    "b": ("B",),
    "d": ("D",),
    "f": ("F",),
    "g": ("G",),
    "h": ("HH",),
    "j": ("Y",),
    "k": ("K",),
    "l": ("L",),
    "m": ("M",),
    "n": ("N",),
    "ŋ": ("NG",),
    "p": ("P",),
    "r": ("R",),
    "s": ("S",),
    "t": ("T",),
    "v": ("V",),
    "w": ("W",),
    "z": ("Z",),
    "ʃ": ("SH",),
    "ʒ": ("ZH",),
    "θ": ("TH",),
    "ð": ("DH",),
}

VOWELS = frozenset({"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UW", "UH", "AH", "ER", "AY", "AW", "OY"})
IPA_SYMBOLS = tuple(sorted(IPA_TO_ARPABET, key=len, reverse=True))


def ipa_to_arpabet(ipa: str) -> list[str]:
    """Convert an IPA string, raising an error rather than guessing."""
    phones: list[str] = []
    position = 0
    while position < len(ipa):
        symbol = next((item for item in IPA_SYMBOLS if ipa.startswith(item, position)), None)
        if symbol is None:
            raise ValueError(f"Unsupported IPA symbol {ipa[position]!r} in {ipa!r}")
        mapped_phones = IPA_TO_ARPABET[symbol]
        # A centering diphthong already contributes ARPABET R.  When the IPA
        # spelling continues with /r/ (e.g. gɛər), it is still one R phone.
        if phones[-1:] == ["R"] and mapped_phones[:1] == ("R",):
            mapped_phones = mapped_phones[1:]
        phones.extend(mapped_phones)
        position += len(symbol)
    return phones


def with_one_before_vowels(phones: list[str]) -> list[str]:
    """Insert a standalone 1 token before every vowel phone."""
    output: list[str] = []
    for phone in phones:
        if phone in VOWELS:
            output.append("1")
        output.append(phone)
    return output


def convert(input_path: Path, output_path: Path) -> int:
    with input_path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.reader(source, delimiter="\t"))

    if not rows or len(rows[0]) < 2:
        raise ValueError("Expected a header and an IPA transcription in column 2.")

    output_rows = [rows[0] + ["ARPABET", "ARPABET_1_BEFORE_VOWEL"]]
    for line_number, row in enumerate(rows[1:], start=2):
        if not any(cell.strip() for cell in row):
            continue
        if len(row) < 2:
            raise ValueError(f"Line {line_number} has fewer than two columns.")
        phones = ipa_to_arpabet(row[1])
        output_rows.append(row + [" ".join(phones), " ".join(with_one_before_vowels(phones))])

    with output_path.open("w", encoding="utf-8", newline="") as destination:
        csv.writer(destination, delimiter="\t", lineterminator="\n").writerows(output_rows)
    return len(output_rows) - 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source tab-delimited data file")
    parser.add_argument("--output", type=Path, required=True, help="Derived tab-delimited data file")
    args = parser.parse_args()
    count = convert(args.input, args.output)
    print(f"Wrote {count} converted rows to {args.output}")


if __name__ == "__main__":
    main()
