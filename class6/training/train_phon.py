"""
Tiny character/phoneme-level transformer for an IPA word list.
Expects a UTF-8 text file with one transcribed word per line.

Usage:
    python train_phon.py --data wordlist.txt
"""
import argparse, math, random, unicodedata
from collections import Counter
import torch, torch.nn as nn, torch.nn.functional as F

p = argparse.ArgumentParser()
p.add_argument("--data", default="wordlist.txt")
p.add_argument("--layers", type=int, default=4)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--dropout", type=float, default=0.2)
p.add_argument("--batch", type=int, default=128)
p.add_argument("--steps", type=int, default=3000)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--max_len", type=int, default=40, help="max tokens per word (longer words are dropped)")
p.add_argument("--seed", type=int, default=1337)
args = p.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("device:", device)

# ---------------------------------------------------------------- tokenizer
TIE = {"\u0361", "\u035c"}   # tie bars: t͡ʃ, d͜ʒ
STRESS = {"ˈ", "ˌ"}          # stress marks stay as their own tokens

def segment(word):
    """Split an IPA string into phoneme-sized units: a base symbol plus any
    following diacritics, modifier letters (ʰ ʲ ʷ ː ...) and tie-bar partners."""
    toks, glue = [], False
    for ch in word:
        cat = unicodedata.category(ch)
        attach = (
            ch in TIE
            or cat.startswith("M")
            or (cat == "Lm" and ch not in STRESS)
        )
        if toks and (glue or attach):
            toks[-1] += ch
            glue = ch in TIE
        else:
            toks.append(ch)
            glue = False
    return toks

# ---------------------------------------------------------------- data
with open(args.data, encoding="utf-8") as f:
    words = [unicodedata.normalize("NFC", l.strip()) for l in f if l.strip()]
words = list(dict.fromkeys(words))  # de-duplicate, keep order
print("entries:", len(words))

seqs = [segment(w) for w in words]
kept = [(w, s) for w, s in zip(words, seqs) if len(s) + 2 <= args.max_len]
print(f"dropped {len(words) - len(kept)} entries longer than {args.max_len - 2} tokens")
words, seqs = zip(*kept)

counts = Counter(t for s in seqs for t in s)
PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
itos = [PAD, BOS, EOS] + sorted(counts)
stoi = {t: i for i, t in enumerate(itos)}
V = len(itos)
print("vocab size:", V)
print("most common:", counts.most_common(12))
print("rare (<=3 occurrences):", [t for t, c in counts.items() if c <= 3])

def encode(s):
    ids = [stoi[BOS]] + [stoi[t] for t in s] + [stoi[EOS]]
    return ids + [0] * (args.max_len - len(ids))

# split BY WORD, never within a word
idx = list(range(len(seqs)))
random.shuffle(idx)
n_val = n_test = len(idx) // 20
val_i, test_i, train_i = idx[:n_val], idx[n_val:n_val + n_test], idx[n_val + n_test:]
mk = lambda ix: torch.tensor([encode(seqs[i]) for i in ix], device=device)
train, val, test = mk(train_i), mk(val_i), mk(test_i)
train_set = {words[i] for i in train_i}
print(f"train {len(train)} / val {len(val)} / test {len(test)}")

def xy(batch):
    x, y = batch[:, :-1], batch[:, 1:].clone()
    y[y == 0] = -100  # don't compute loss on padding
    return x, y

# ---------------------------------------------------------------- model
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
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=s.p if s.training else 0.0)
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
        s.head.weight = s.tok.weight  # weight tying

    def forward(s, x, y=None):
        T = x.shape[1]
        h = s.drop(s.tok(x) + s.pos(torch.arange(T, device=x.device)))
        for b in s.blocks:
            h = b(h)
        logits = s.head(s.ln(h))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=-100) if y is not None else None
        return logits, loss

model = GPT(V, args.dim, args.layers, args.heads, args.dropout, args.max_len).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

@torch.no_grad()
def evaluate(data):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(data), 512):
        x, y = xy(data[i:i + 512])
        _, loss = model(x, y)
        k = (y != -100).sum().item()
        tot += loss.item() * k
        n += k
    model.train()
    return tot / n

# ---------------------------------------------------------------- train
best, best_state = 1e9, None
for step in range(args.steps + 1):
    lr = args.lr * 0.5 * (1 + math.cos(math.pi * step / args.steps))
    for g in opt.param_groups:
        g["lr"] = lr
    x, y = xy(train[torch.randint(len(train), (args.batch,), device=device)])
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 250 == 0:
        v = evaluate(val)
        print(f"step {step:5d} | train {loss.item():.3f} | val {v:.3f} (ppl {math.exp(v):.2f})")
        if v < best:
            best = v
            best_state = {k: t.clone() for k, t in model.state_dict().items()}

model.load_state_dict(best_state)
t = evaluate(test)
print(f"\nbest val {best:.3f} | test {t:.3f} (ppl {math.exp(t):.2f}, {t / math.log(2):.2f} bits/token)")
torch.save({"model": best_state, "itos": itos, "args": vars(args)}, "phon_model.pt")

# ---------------------------------------------------------------- sample
@torch.no_grad()
def sample(temp=1.0):
    model.eval()
    ids = [stoi[BOS]]
    while len(ids) < args.max_len:
        logits, _ = model(torch.tensor([ids], device=device))
        probs = F.softmax(logits[0, -1] / temp, dim=-1)
        probs[0] = 0  # never emit pad
        nxt = torch.multinomial(probs, 1).item()
        if nxt == stoi[EOS]:
            break
        ids.append(nxt)
    return "".join(itos[i] for i in ids[1:])

print("\nsamples (* = copied from training set):")
for _ in range(25):
    w = sample()
    print(" ", w, "*" if w in train_set else "")

# ---------------------------------------------------------------- probe embeddings
E = F.normalize(model.tok.weight.detach().cpu(), dim=1)
print("\nnearest neighbours in embedding space (most frequent tokens):")
for tkn, _ in counts.most_common(15):
    sims = E @ E[stoi[tkn]]
    nn_ = [itos[j] for j in sims.argsort(descending=True)[1:5].tolist()]
    print(f"  {tkn}: {' '.join(nn_)}")
