"""
Convert Arpabet (CMU style, e.g. "HH AH0 L OW1") to IPA.

Examples:
    python arpa2ipa.py "HH AH0 L OW1"              # -> həlˈoʊ
    python arpa2ipa.py --file new_words.txt         # one IPA word per line
    python arpa2ipa.py --file new_words.txt --out new_words_ipa.txt
    cat new_words.txt | python arpa2ipa.py          # read from stdin
    python arpa2ipa.py "CH ER1 CH" --tie            # -> t͡ʃˈɝt͡ʃ
    python arpa2ipa.py "CH ER1 CH" --space          # -> tʃ ˈɝ tʃ
"""
import argparse, re, sys

CONS = {
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "F": "f", "G": "ɡ", "HH": "h",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ", "P": "p",
    "R": "ɹ", "S": "s", "SH": "ʃ", "T": "t", "TH": "θ", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}
VOWELS = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "EH": "ɛ", "ER": "ɝ", "EY": "eɪ", "IH": "ɪ", "IY": "i", "OW": "oʊ",
    "OY": "ɔɪ", "UH": "ʊ", "UW": "u",
}
UNSTRESSED = {"AH": "ə", "ER": "ɚ"}     # AH0 -> ə, ER0 -> ɚ
STRESS = {"1": "ˈ", "2": "ˌ"}           # primary, secondary
TIE = {"tʃ": "t͡ʃ", "dʒ": "d͡ʒ"}

def arpa_to_ipa(arpa, tie=False, space=False):
    """Return (ipa_string, list_of_unrecognised_tokens)."""
    out, bad = [], []
    for tok in arpa.split():
        m = re.fullmatch(r"([A-Za-z]+)([012])?", tok)
        ph, st = (m.group(1).upper(), m.group(2)) if m else (None, None)
        if ph in VOWELS:
            v = VOWELS[ph] if st in STRESS or ph not in UNSTRESSED else UNSTRESSED[ph]
            out.append(STRESS.get(st, "") + v)       # stress mark goes right before the vowel
        elif ph in CONS:
            c = CONS[ph]
            out.append(TIE.get(c, c) if tie else c)
        else:
            bad.append(tok)
    return (" " if space else "").join(out), bad

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("word", nargs="?", help='one Arpabet word in quotes, e.g. "K AE1 T"')
    p.add_argument("--file", help="text file with one Arpabet word per line (default: stdin)")
    p.add_argument("--out", help="write the IPA to this file, one word per line")
    p.add_argument("--tie", action="store_true", help="use tie bars for affricates: t͡ʃ d͡ʒ")
    p.add_argument("--space", action="store_true", help="put a space between phonemes")
    args = p.parse_args()

    if args.word:
        lines = [args.word]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            lines = [l.strip().rstrip("*").strip() for l in f if l.strip()]
    else:
        lines = [l.strip().rstrip("*").strip() for l in sys.stdin if l.strip()]

    results = []
    for line in lines:
        ipa, bad = arpa_to_ipa(line, args.tie, args.space)
        if bad:
            print(f"warning: '{line}' has unrecognised symbols {bad}", file=sys.stderr)
        results.append(ipa)

    text = "\n".join(results) + "\n"
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {len(results)} words to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
