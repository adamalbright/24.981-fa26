"""
Speak Arpabet transcriptions (CMU style, e.g. "HH AH0 L OW1") with espeak-ng.

Examples:
    python speak.py "HH AH0 L OW1"                 # play one word
    python speak.py --file new_words.txt           # play every line of a file
    python speak.py --file new_words.txt --wav out # write out/0001.wav, out/0002.wav, ...

Needs espeak-ng:  brew install espeak-ng   (macOS)   |   sudo apt install espeak-ng
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, wave

# Arpabet -> espeak-ng phoneme mnemonics (American English voice)
CONS = {
    "B": "b", "CH": "tS", "D": "d", "DH": "D", "F": "f", "G": "g", "HH": "h",
    "JH": "dZ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "N", "P": "p",
    "R": "r", "S": "s", "SH": "S", "T": "t", "TH": "T", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "Z",
}
VOWELS = {
    "AA": "A:", "AE": "a", "AH": "V", "AO": "O:", "AW": "aU", "AY": "aI",
    "EH": "E", "ER": "3:", "EY": "eI", "IH": "I", "IY": "i:", "OW": "oU",
    "OY": "OI", "UH": "U", "UW": "u:",
}
UNSTRESSED = {"AH": "@", "ER": "3"}  # AH0 -> schwa, ER0 -> unstressed r-coloured vowel
STRESS = {"1": "'", "2": ","}

def to_espeak(arpa):
    out, bad = [], []
    for tok in arpa.split():
        m = re.fullmatch(r"([A-Za-z]+)([012])?", tok)
        if not m:
            bad.append(tok); continue
        ph, st = m.group(1).upper(), m.group(2)
        if ph in VOWELS:
            if st in STRESS:
                out.append(STRESS[st] + VOWELS[ph])
            else:
                out.append(UNSTRESSED.get(ph, VOWELS[ph]))
        elif ph in CONS:
            out.append(CONS[ph])
        else:
            bad.append(tok)
    return "".join(out), bad

p = argparse.ArgumentParser()
p.add_argument("word", nargs="?", help='one Arpabet word in quotes, e.g. "K AE1 T"')
p.add_argument("--file", help="text file with one Arpabet word per line")
p.add_argument("--wav", help="directory to write numbered .wav files instead of playing")
p.add_argument("--speed", type=int, default=130, help="words per minute (espeak default 175)")
p.add_argument("--voice", default="en-us")
p.add_argument("--pad", type=float, default=0.4, help="seconds of silence added before and after each word")
p.add_argument("--pitch", type=int, default=50, help="0-99 (espeak default 50)")
p.add_argument("--amp", type=int, default=100, help="volume 0-200 (espeak default 100)")
args = p.parse_args()

if not shutil.which("espeak-ng"):
    sys.exit("espeak-ng not found. Install it first (brew install espeak-ng).")

lines = [args.word] if args.word else []
if args.file:
    with open(args.file, encoding="utf-8") as f:
        lines += [l.strip().rstrip("*").strip() for l in f if l.strip()]
if not lines:
    p.error("give a word or --file")
if args.wav:
    os.makedirs(args.wav, exist_ok=True)

def pad_wav(path, seconds):
    """Add silence before and after, so players don't clip the word."""
    with wave.open(path, "rb") as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    sil = b"\x00" * (int(params.framerate * seconds) * params.sampwidth * params.nchannels)
    with wave.open(path, "wb") as w:
        w.setparams(params)
        w.writeframes(sil + frames + sil)

def play(path):
    for cmd in (["afplay"], ["paplay"], ["aplay", "-q"],
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            subprocess.run(cmd + [path], check=True)
            return
    sys.exit("No audio player found (looked for afplay, paplay, aplay, ffplay). "
             "Use --wav DIR to save files instead.")

tmpdir = None if args.wav else tempfile.mkdtemp()
for i, line in enumerate(lines, 1):
    ph, bad = to_espeak(line)
    if bad:
        print(f"skipping '{line}': unrecognised symbols {bad}")
        continue
    path = os.path.join(args.wav or tmpdir, f"{i:04d}.wav")
    # Always render to a file (never straight to the sound card), then pad and play it.
    subprocess.run(["espeak-ng", "-v", args.voice, "-s", str(args.speed),
                    "-p", str(args.pitch), "-a", str(args.amp), f"[[{ph}]]", "-w", path],
                   check=True)
    pad_wav(path, args.pad)
    print(f"{i:4d}  {line:30s} -> {ph}")
    if not args.wav:
        play(path)
        os.remove(path)
