"""
Generate more words from a model trained with train_phon.py.

Examples:
    python sample.py                                  # 50 words
    python sample.py -n 500 --temp 0.8                # more conservative
    python sample.py -n 500 --novel --data wordlist.txt --out new_words.txt
"""
import argparse, sys, unicodedata
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="phon_model.pt")
p.add_argument("-n", type=int, default=50, help="number of words to generate")
p.add_argument("--temp", type=float, default=1.0, help="<1 safer/more typical, >1 more adventurous")
p.add_argument("--top_k", type=int, default=0, help="sample only from the k most likely tokens (0 = off)")
p.add_argument("--novel", action="store_true", help="keep only words NOT in the training data (needs --data)")
p.add_argument("--data", default=None, help="the original word list, used by --novel and to mark copies")
p.add_argument("--unique", action="store_true", help="drop duplicate samples")
p.add_argument("--out", default=None, help="write the words to this file, one per line")
p.add_argument("--seed", type=int, default=None)
p.add_argument("--raw", action="store_true",
               help="print samples exactly as generated (stress digits as separate symbols)")
p.add_argument("--keep_bad", action="store_true",
               help="keep samples whose stress digits don't sit right before a vowel")
args = p.parse_args()

# Arpabet vowels; a stress digit (0/1/2) generated BEFORE a vowel is moved onto it:
#   "B 1 AW T"  ->  "B AW1 T"
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
          "OW", "OY", "UH", "UW"}

def merge_stress(s):
    """Return (CMU-style string, well_formed)."""
    out, pending, ok = [], None, True
    for t in s.split():
        if t in ("0", "1", "2"):
            ok &= pending is None          # two stress marks in a row
            pending = t
        elif t in VOWELS:
            out.append(t + (pending or ""))
            pending = None
        else:
            ok &= pending is None          # stress mark not followed by a vowel
            pending = None
            out.append(t)
    ok &= pending is None                  # stress mark at the very end
    return " ".join(out), ok

if args.seed is not None:
    torch.manual_seed(args.seed)
device = "mps" if torch.backends.mps.is_available() else "cpu"

ck = torch.load(args.ckpt, map_location=device)
itos, a = ck["itos"], ck["args"]
stoi = {t: i for i, t in enumerate(itos)}
BOS, EOS = stoi["<bos>"], stoi["<eos>"]
V, max_len = len(itos), a["max_len"]

# ---- same model definition as train_phon.py (must match to load the weights)
class Block(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__()
        s.h, s.p = h, drop
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv, s.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.drop = nn.Dropout(drop)
    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=s.p if s.training else 0.0)
        x = x + s.drop(s.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + s.drop(s.mlp(s.ln2(x)))

class GPT(nn.Module):
    def __init__(s, V, d, L, h, drop, max_len):
        super().__init__()
        s.tok = nn.Embedding(V, d)
        s.pos = nn.Embedding(max_len, d)
        s.drop = nn.Dropout(drop)
        s.blocks = nn.ModuleList(Block(d, h, drop) for _ in range(L))
        s.ln = nn.LayerNorm(d)
        s.head = nn.Linear(d, V, bias=False)
        s.head.weight = s.tok.weight
    def forward(s, x):
        T = x.shape[1]
        h = s.drop(s.tok(x) + s.pos(torch.arange(T, device=x.device)))
        for b in s.blocks:
            h = b(h)
        return s.head(s.ln(h))

# Whole-symbol vocabularies (e.g. "AH", "NG", "1") contain no space token, so the
# symbols must be joined with spaces. Character-level vocabularies that include
# " " as a token already carry their own spaces, so join with nothing.
SEP = "" if " " in itos else " "
print(f"vocab: {V} tokens, joining with {'nothing' if SEP == '' else 'spaces'}", file=sys.stderr)

model = GPT(V, a["dim"], a["layers"], a["heads"], a["dropout"], max_len).to(device)
model.load_state_dict(ck["model"])
model.eval()

known = set()
if args.data:
    with open(args.data, encoding="utf-8") as f:
        known = {" ".join(unicodedata.normalize("NFC", l).split()) for l in f if l.strip()}
if args.novel and not known:
    p.error("--novel needs --data pointing at your original word list")

@torch.no_grad()
def sample():
    ids = [BOS]
    while len(ids) < max_len:
        logits = model(torch.tensor([ids], device=device))[0, -1] / args.temp
        logits[0] = logits[BOS] = float("-inf")        # never emit <pad> or <bos>
        if args.top_k:
            kth = torch.topk(logits, args.top_k).values[-1]
            logits[logits < kth] = float("-inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
        if nxt == EOS:
            break
        ids.append(nxt)
    return SEP.join(itos[i] for i in ids[1:])

out, seen, tries, malformed = [], set(), 0, 0   # out holds (text, is_copy)
while len(out) < args.n and tries < args.n * 50:
    tries += 1
    raw = " ".join(sample().split())               # normalise spacing
    if not raw or (args.novel and raw in known) or (args.unique and raw in seen):
        continue
    text, ok = (raw, True) if args.raw else merge_stress(raw)
    if not ok:
        malformed += 1
        if not args.keep_bad:
            continue
    seen.add(raw)
    out.append((text, raw in known))

for text, copy in out:
    print(text + ("  *" if copy else ""))
if known:
    print(f"\n{sum(c for _, c in out)}/{len(out)} are copies of the training data (*)")
if malformed:
    print(f"{malformed} samples had a stress digit not directly before a vowel "
          f"({'kept' if args.keep_bad else 'discarded'})")
if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(t for t, _ in out) + "\n")
    print(f"wrote {len(out)} words to {args.out}")
