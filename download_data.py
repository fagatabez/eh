# download_data.py
import gzip
import shutil
import os

os.environ["HF_DATASETS_CACHE"] = "./hf_cache"
os.environ["HF_HOME"] = "./hf_cache"

from datasets import load_dataset

DATA_GZ  = "training_data.txt.gz"
DATA_TXT = "training_data.txt"

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

MODE = "add"

DATASETS_TO_DOWNLOAD = {
    "assistant":    True,   # real LLM-style conversations
    "coding":       True,   # write and explain code
    "explanation":  True,   # explain concepts in detail
    "conversation": True,   # casual chat
}

SIZES = {
    "assistant":    20000,
    "coding":       20000,
    "explanation":  20000,
    "conversation": 20000,
}

# ═══════════════════════════════════════════════════

existing_lines = []

if os.path.exists(DATA_GZ):
    if MODE == "replace":
        print("Replacing old data...")
        os.remove(DATA_GZ)
    elif MODE == "add":
        print("Loading existing data to merge...")
        with gzip.open(DATA_GZ, "rt", encoding="utf-8") as f:
            content = f.read()
        existing_lines = [l for l in content.strip().split("\n\n") if l.strip()]
        print(f"Loaded {len(existing_lines):,} existing pairs")

seen      = set(l.split("\n")[0].lower() for l in existing_lines if l)
all_lines = list(existing_lines)

def add_pair(q, a):
    q = q.strip()
    a = a.strip()
    if not q or not a:
        return False
    if len(a) < 20:
        return False
    key = q.lower()[:80]
    if key in seen:
        return False
    seen.add(key)
    all_lines.append(f"Question: {q}\nAnswer: {a}")
    return True

# ── Dataset 1: OpenAssistant — real LLM conversations ────────
if DATASETS_TO_DOWNLOAD["assistant"]:
    print(f"\nDownloading OpenAssistant (real LLM conversations)...")
    count = 0
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        # build a map of id -> text
        id_to_text = {}
        for item in ds:
            id_to_text[item["message_id"]] = item

        for item in ds:
            if item["role"] != "assistant":
                continue
            a = item["text"].strip()
            # find parent question
            parent_id = item.get("parent_id")
            if parent_id and parent_id in id_to_text:
                q = id_to_text[parent_id]["text"].strip()
            else:
                continue
            if add_pair(q, a):
                count += 1
            if count >= SIZES["assistant"]:
                break
        print(f"OpenAssistant done — added {count:,} pairs")
    except Exception as e:
        print(f"OpenAssistant failed: {e}")

# replace the CodeAlpaca and ELI5 sections with these working ones

# ── Dataset 2: Code instructions ─────────────────────────────
if DATASETS_TO_DOWNLOAD["coding"]:
    print(f"\nDownloading coding dataset...")
    count = 0
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for i, item in enumerate(ds):
            q = item.get("instruction", "").strip()
            a = item.get("output", "").strip()
            if add_pair(q, a):
                count += 1
            if count >= SIZES["coding"]:
                break
        print(f"Coding done — added {count:,} pairs")
    except Exception as e:
        print(f"Coding failed: {e}")

# ── Dataset 3: ELI5 explanations ─────────────────────────────
if DATASETS_TO_DOWNLOAD["explanation"]:
    print(f"\nDownloading explanations dataset...")
    count = 0
    try:
        ds = load_dataset("duorc", "SelfRC", split="train")
        for i, item in enumerate(ds):
            q = item.get("question", "").strip()
            answers = item.get("answers", [])
            if answers:
                a = max(answers, key=len).strip()
                if add_pair(q, a):
                    count += 1
            if count >= SIZES["explanation"]:
                break
        print(f"Explanations done — added {count:,} pairs")
    except Exception as e:
        print(f"Explanations failed: {e}")

# ── Dataset 4: Alpaca — general assistant instructions ───────
if DATASETS_TO_DOWNLOAD["conversation"]:
    print(f"\nDownloading Alpaca (general assistant)...")
    count = 0
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for i, item in enumerate(ds):
            q = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            a   = item.get("output", "").strip()
            if inp:
                q = f"{q}\n{inp}"
            if add_pair(q, a):
                count += 1
            if count >= SIZES["conversation"]:
                break
        print(f"Alpaca done — added {count:,} pairs")
    except Exception as e:
        print(f"Alpaca failed: {e}")

# ── Save ─────────────────────────────────────────────────────
print(f"\nTotal Q&A pairs: {len(all_lines):,}")
text = "\n\n".join(all_lines)

print("Saving...")
with open(DATA_TXT, "w", encoding="utf-8") as f:
    f.write(text)

print("Compressing...")
with open(DATA_TXT, "rb") as f_in:
    with gzip.open(DATA_GZ, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

original   = os.path.getsize(DATA_TXT)
compressed = os.path.getsize(DATA_GZ)
saving     = 100 - (compressed / original * 100)
print(f"Original:   {original/1024/1024:.1f} MB")
print(f"Compressed: {compressed/1024/1024:.1f} MB")
print(f"Saved:      {saving:.0f}% space")

os.remove(DATA_TXT)

if os.path.exists("./hf_cache"):
    shutil.rmtree("./hf_cache")
    print("Cleared HuggingFace cache")

print("\nDone! Run train.py next.")