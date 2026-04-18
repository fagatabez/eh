# chat.py — run on your PC after training
import torch, json, os, re
from model import MyAI, Config
from tokenizer import Tokenizer

# ═══════════════════════════════════════════════════
#   INFERENCE SETTINGS FILE
#   Created automatically on first run.
#   Edit chat_settings.json to control what the AI uses.
#
#   use_trained_chars: how many chars worth of training to use.
#       "all"   → use the full model as-is (default)
#       "500k"  → mentally cap to 500k chars of learning
#                 (scales temperature up so undertrained = more cautious)
#       number  → same as above but as raw int
#
#   use_full: true/false — shortcut for "all"
#   temperature: 0.1 (focused) → 1.5 (creative/random)
#   top_k: vocabulary sampling width (20=focused, 80=varied)
#   max_new_tokens: max words to generate per reply
# ═══════════════════════════════════════════════════

SETTINGS_FILE = "chat_settings.json"

DEFAULT_SETTINGS = {
    "use_full":          True,
    "use_trained_chars": "all",
    "temperature":       0.8,
    "top_k":             40,
    "max_new_tokens":    150,
    "_help": {
        "use_trained_chars": "how much training to trust: 'all', '1m', '500k', or a number",
        "use_full":          "true = use everything trained (overrides use_trained_chars)",
        "temperature":       "0.1=focused, 0.8=balanced, 1.5=creative",
        "top_k":             "20=focused, 40=balanced, 80=varied",
        "max_new_tokens":    "max words per reply",
    }
}

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                saved = json.load(f)
            # Merge with defaults so new keys always appear
            merged = dict(DEFAULT_SETTINGS)
            merged.update({k: v for k, v in saved.items() if k != "_help"})
            return merged
        except Exception as e:
            print(f"[settings] Could not read {SETTINGS_FILE}: {e} — using defaults")
    return dict(DEFAULT_SETTINGS)

def save_settings(s):
    out = {k: v for k, v in s.items()}
    with open(SETTINGS_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[settings] Saved to {SETTINGS_FILE}")

def parse_chars(val, trained_chars):
    """Parse use_trained_chars value → int number of chars (or None = all)."""
    if val is None or str(val).lower() in ("all", "true", ""):
        return None   # use all
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().lower().replace(",", "").replace("_", "")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try: return int(float(s[:-1]) * mult)
            except: break
    try: return int(s)
    except: return None

def chars_to_temperature_boost(target_chars, trained_chars):
    """
    If the model was only trained on target_chars out of trained_chars total,
    boost temperature slightly to reflect lower confidence.
    Returns a multiplier to apply on top of base temperature.
    """
    if target_chars is None or trained_chars == 0:
        return 1.0
    ratio = min(target_chars, trained_chars) / max(trained_chars, 1)
    # ratio 1.0 (fully trained) → boost 1.0; ratio 0.1 → boost 1.4
    return 1.0 + (1.0 - ratio) * 0.5

# ═══════════════════════════════════════════════════
#   MODEL LOADING
# ═══════════════════════════════════════════════════

def load_model(path="myai.pt"):
    data = torch.load(path, map_location="cpu")
    cfg  = Config()
    for k, v in data["config"].items():
        setattr(cfg, k, v)
    model = MyAI(cfg)
    model.load_state_dict(data["model"])
    model.eval()
    return model, cfg, data

# ═══════════════════════════════════════════════════
#   OUTPUT CLEANING
# ═══════════════════════════════════════════════════

def clean_output(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)
    lines = []
    for line in text.split("\n"):
        if not line.strip(): continue
        alpha = sum(c.isalpha() or c.isspace() for c in line)
        if len(line) == 0 or alpha / len(line) >= 0.4:
            lines.append(line)
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text).strip()
    words = [w for w in text.split() if len(w) > 1 and w.isalpha()]
    if len(words) < 3:
        return None
    return text

# ═══════════════════════════════════════════════════
#   GENERATION
# ═══════════════════════════════════════════════════

def generate(model, cfg, tok, prompt, max_new=150, temperature=0.8, top_k=40):
    formatted = f"Question: {prompt}\nAnswer:"
    ids       = tok.encode(formatted)
    x         = torch.tensor([ids])
    eos_id    = tok.special.get("<eos>", 3)
    pad_id    = tok.special.get("<pad>", 0)
    unk_id    = tok.special.get("<unk>", 1)
    unk_streak = 0
    MAX_UNK    = 5

    with torch.no_grad():
        for _ in range(max_new):
            x_in   = x[:, -cfg.seq_len+1:]
            logits = model(x_in)[0, -1]
            logits[pad_id] = float("-inf")
            logits = logits / max(temperature, 1e-6)
            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[-1]] = float("-inf")
            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == eos_id: break
            if next_token == unk_id:
                unk_streak += 1
                if unk_streak >= MAX_UNK: break
            else:
                unk_streak = 0
            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)
            decoded_so_far = tok.decode(x[0, len(ids):].tolist())
            if "question:" in decoded_so_far.lower(): break

    raw = tok.decode(x[0, len(ids):].tolist())
    if "question:" in raw.lower():
        raw = raw[:raw.lower().index("question:")].strip()
    return clean_output(raw)

# ═══════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════

# ── Load settings ──────────────────────────────────
settings = load_settings()

# Create settings file on first run so user can find and edit it
if not os.path.exists(SETTINGS_FILE):
    save_settings(settings)
    print(f"\n[settings] Created {SETTINGS_FILE} — edit it to control chat behaviour.")

# ── Load model ─────────────────────────────────────
tok = Tokenizer()
tok.load("tokenizer.json")
model, cfg, ckpt_data = load_model("myai.pt")

# ── Read training info from checkpoint ─────────────
trained_chars = ckpt_data.get("trained_chars", 0)
best_loss     = ckpt_data.get("best_loss", None)
epoch         = ckpt_data.get("epoch", 0)

# Fallback: try reading from .training_complete.json
if trained_chars == 0 and os.path.exists(".training_complete.json"):
    try:
        with open(".training_complete.json") as f:
            tc = json.load(f)
        trained_chars = tc.get("chars_trained", 0)
        if best_loss is None:
            best_loss = tc.get("best_loss", None)
    except Exception:
        pass

# ── Apply inference settings ───────────────────────
use_full   = settings.get("use_full", True)
target_raw = settings.get("use_trained_chars", "all")

if use_full or str(target_raw).lower() in ("all", "true", ""):
    target_chars = None   # use full model
else:
    target_chars = parse_chars(target_raw, trained_chars)

base_temp  = float(settings.get("temperature", 0.8))
top_k      = int(settings.get("top_k", 40))
max_new    = int(settings.get("max_new_tokens", 150))

# Adjust temperature based on how much training we're "trusting"
temp_boost = chars_to_temperature_boost(target_chars, trained_chars)
temperature = base_temp * temp_boost

# ── Status printout ────────────────────────────────
print(f"\nMyAI Chat")
print("=" * 42)

if trained_chars > 0:
    def _fmt(n):
        if n >= 1_000_000: return f"{n/1_000_000:.1f}m"
        if n >= 1_000:     return f"{n/1_000:.0f}k"
        return str(n)
    print(f"  Trained on : {_fmt(trained_chars)} chars")
    if target_chars is not None:
        pct = min(target_chars, trained_chars) / trained_chars * 100
        print(f"  Using      : {_fmt(target_chars)} chars ({pct:.0f}% of trained)")
        print(f"  Temperature: {temperature:.2f} (boosted from {base_temp} — less training = wider sampling)")
    else:
        print(f"  Using      : all trained data")
        print(f"  Temperature: {temperature:.2f}")
else:
    print(f"  Trained on : unknown (no .training_complete.json found)")
    print(f"  Temperature: {temperature:.2f}")

if best_loss is not None:
    quality = "good" if best_loss < 1.5 else "fair" if best_loss < 3.0 else "still learning"
    print(f"  Best loss  : {best_loss:.4f} ({quality})")
    if best_loss > 4.0:
        print(f"\n  ⚠ Loss is still high — answers may be poor.")
        print(f"    Keep training to improve quality.")
elif epoch < 5:
    print(f"\n  Note: only {epoch} epoch(s) trained. Answers may be poor until 10+ epochs.")

print(f"\n  Settings file: {SETTINGS_FILE}")
print(f"  Edit it to change temperature, top_k, or use_trained_chars.")
print(f"\nType 'quit' to exit, 'settings' to view current settings.\n")

while True:
    try:
        prompt = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!"); break

    if not prompt:
        continue
    if prompt.lower() in ("quit", "exit", "q"):
        break
    if prompt.lower() == "settings":
        print(f"\nCurrent settings ({SETTINGS_FILE}):")
        for k, v in settings.items():
            if k != "_help":
                print(f"  {k}: {v}")
        print()
        continue

    result = generate(model, cfg, tok, prompt,
                      max_new=max_new, temperature=temperature, top_k=top_k)

    if result is None:
        print("AI: I haven't learned enough yet to answer that. Keep training!\n")
    else:
        print(f"AI: {result}\n")

