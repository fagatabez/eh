# chat.py — run on your PC after training
import torch
from model import MyAI, Config
from tokenizer import Tokenizer

def load_model(path="myai.pt"):
    data = torch.load(path, map_location="cpu")
    cfg  = Config()
    for k, v in data["config"].items():
        setattr(cfg, k, v)
    model = MyAI(cfg)
    model.load_state_dict(data["model"])
    model.eval()
    print(f"Loaded: {data.get('system','').strip()[:60]}...")
    return model, cfg

def clean_output(text: str) -> str:
    """
    Remove garbage that an undertrained model produces:
      - <unk> and any other <...> tokens
      - long runs of repeated characters (e.g. "..........." or "lllllll")
      - isolated punctuation clutter
      - double spaces
    Returns cleaned text, or a fallback message if nothing usable remains.
    """
    import re

    # remove special tokens
    text = re.sub(r"<[^>]+>", "", text)

    # remove runs of 4+ identical chars (e.g. "......" or "aaaaaaa")
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)

    # remove lines that are >50% non-alphanumeric (garbage lines)
    lines = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        alpha = sum(c.isalpha() or c.isspace() for c in line)
        if len(line) == 0 or alpha / len(line) >= 0.4:
            lines.append(line)
    text = " ".join(lines)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # if less than 3 real words remain, the model hasn't learned yet
    words = [w for w in text.split() if len(w) > 1 and w.isalpha()]
    if len(words) < 3:
        return None   # caller will use fallback

    return text

def generate(model, cfg, tok, prompt, max_new=150, temperature=0.8, top_k=40):
    formatted  = f"Question: {prompt}\nAnswer:"
    ids        = tok.encode(formatted)
    x          = torch.tensor([ids])
    eos_id     = tok.special.get("<eos>", 3)
    pad_id     = tok.special.get("<pad>", 0)
    unk_id     = tok.special.get("<unk>", 1)

    # count consecutive <unk> tokens — stop if the model is just spitting garbage
    unk_streak = 0
    MAX_UNK    = 5

    with torch.no_grad():
        for _ in range(max_new):
            x_in   = x[:, -cfg.seq_len+1:]
            logits = model(x_in)[0, -1]

            # mask out pad token to avoid spitting padding
            logits[pad_id] = float("-inf")

            # temperature
            logits = logits / max(temperature, 1e-6)

            # top-k sampling
            if top_k > 0:
                values, _  = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[-1]] = float("-inf")

            probs      = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            if next_token == eos_id:
                break

            if next_token == unk_id:
                unk_streak += 1
                if unk_streak >= MAX_UNK:
                    break   # model is lost — stop
            else:
                unk_streak = 0

            x = torch.cat([x, torch.tensor([[next_token]])], dim=1)

            # stop if the model started writing a new "Question:"
            decoded_so_far = tok.decode(x[0, len(ids):].tolist())
            if "question:" in decoded_so_far.lower():
                break

    raw    = tok.decode(x[0, len(ids):].tolist())
    # cut off any "Question: ..." the model started writing
    if "question:" in raw.lower():
        raw = raw[:raw.lower().index("question:")].strip()

    cleaned = clean_output(raw)
    return cleaned

# ── Main ──────────────────────────────────────────

tok = Tokenizer()
tok.load("tokenizer.json")
model, cfg = load_model("myai.pt")

training_notice = ""
# quick check: if loss is still high the model will be bad — warn the user
try:
    ckpt = torch.load("myai.pt", map_location="cpu")
    # we don't store loss in checkpoint, but we can infer from epoch count
    epoch = ckpt.get("epoch", 0)
    if epoch < 5:
        training_notice = (
            f"\n[Note: model has only trained for {epoch} epoch(s). "
            "Answers may be poor until ~10+ epochs.]\n"
        )
except:
    pass

print(f"\nYour AI is ready!{training_notice}Type 'quit' to exit.\n")

while True:
    prompt = input("You: ").strip()
    if not prompt:
        continue
    if prompt.lower() in ("quit", "exit", "q"):
        break

    result = generate(model, cfg, tok, prompt)

    if result is None:
        print("AI: I haven't learned enough yet to answer that. Keep training!\n")
    else:
        print(f"AI: {result}\n")
