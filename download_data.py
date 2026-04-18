# download_data.py — chunked download with crash resume
# ── Auto-install datasets if missing ──────────────────────────────────────────
import sys, subprocess

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"[setup] installing {pkg}...")
        for flags in [["--break-system-packages", "-q"], ["-q"]]:
            r = subprocess.run([sys.executable, "-m", "pip", "install", pkg] + flags,
                               capture_output=True, text=True)
            if r.returncode == 0: break
        print(f"[setup] {pkg} OK")

_ensure("datasets")
# ──────────────────────────────────────────────────────────────────────────────

import gzip, json, os, shutil, time, re
from datasets import load_dataset

os.environ["HF_DATASETS_CACHE"] = "./hf_cache"
os.environ["HF_HOME"]           = "./hf_cache"

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

DATA_GZ       = "training_data.txt.gz"
PROGRESS_FILE = ".dl_progress.json"

# Flush to .gz after accumulating this many raw characters (~500 MB)
CHUNK_CHARS = 500_000_000

# 0 = download all available; positive number = max pairs per dataset
SIZES = {
    "assistant":    20_000,
    "coding":       20_000,
    "explanation":  20_000,
    "conversation": 20_000,
}

DATASETS_TO_DOWNLOAD = {
    "assistant":    True,
    "coding":       True,
    "explanation":  True,
    "conversation": True,
}

# ═══════════════════════════════════════════════════
#   PROGRESS / CRASH RESUME
# ═══════════════════════════════════════════════════

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"  [progress] Could not read progress file: {e} — starting fresh")
    return {"completed": [], "current": None, "current_index": 0}

def save_progress(prog):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(prog, f, indent=2)

def clear_progress():
    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

# ═══════════════════════════════════════════════════
#   GZ VALIDATION AND SEEN-SET REBUILD
# ═══════════════════════════════════════════════════

def validate_gz(path):
    """
    Check the .gz file is not corrupted by trying to read through it fully.
    Returns (ok: bool, pairs: int, seen: set, char_count: int).
    """
    if not os.path.exists(path):
        return True, 0, set(), 0   # doesn't exist = not corrupted

    print(f"  Validating {path} ...", end="", flush=True)
    try:
        seen   = set()
        pairs  = 0
        chars  = 0
        buf    = ""

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(4 * 1024 * 1024)   # 4 MB at a time
                if not chunk:
                    break
                chars += len(chunk)
                buf   += chunk
                # extract complete pairs (split on \n\n)
                parts  = buf.split("\n\n")
                buf    = parts[-1]               # keep incomplete trailing part
                for p in parts[:-1]:
                    p = p.strip()
                    if not p: continue
                    key = p.split("\n")[0].lower()[:80]
                    seen.add(key)
                    pairs += 1

        # handle last part
        if buf.strip():
            key = buf.strip().split("\n")[0].lower()[:80]
            seen.add(key)
            pairs += 1

        print(f" OK — {pairs:,} pairs, {chars/1024/1024:.0f} MB")
        return True, pairs, seen, chars
    except Exception as e:
        print(f" CORRUPTED ({e})")
        return False, 0, set(), 0

# ═══════════════════════════════════════════════════
#   BUFFER → GZ FLUSH
# ═══════════════════════════════════════════════════

_gz_has_content = False   # tracks whether DATA_GZ already has content

def flush_buffer(buffer_pairs):
    """
    Compress-append the buffered pairs to DATA_GZ.
    Uses Python gzip append mode — creates a new gzip member in the same file.
    Standard decompressors (Python gzip, zcat, etc.) handle multi-member .gz fine.
    """
    global _gz_has_content
    if not buffer_pairs:
        return

    text = "\n\n".join(buffer_pairs)
    if _gz_has_content:
        text = "\n\n" + text   # separator from previous content

    mode = "ab" if _gz_has_content else "wb"
    with gzip.open(DATA_GZ, mode) as f:
        f.write(text.encode("utf-8"))

    _gz_has_content = True
    size_mb = os.path.getsize(DATA_GZ) / 1024 / 1024
    print(f"  ✓ Flushed {len(buffer_pairs):,} pairs — {DATA_GZ}: {size_mb:.0f} MB total")

# ═══════════════════════════════════════════════════
#   PAIR BUILDER
# ═══════════════════════════════════════════════════

def make_pair(q, a):
    return f"Question: {q.strip()}\nAnswer: {a.strip()}"

# ═══════════════════════════════════════════════════
#   DATASET DOWNLOADERS
# ═══════════════════════════════════════════════════

def dl_assistant(seen, max_pairs, resume_index=0):
    """OpenAssistant/oasst1"""
    print(f"\n[1/4] OpenAssistant (real LLM conversations) — resume_index={resume_index}")
    pairs = []; count = 0
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train")
        id_to = {item["message_id"]: item for item in ds}

        for idx, item in enumerate(ds):
            if idx < resume_index: continue
            if item["role"] != "assistant": continue
            pid = item.get("parent_id")
            if not pid or pid not in id_to: continue
            q = id_to[pid]["text"].strip()
            a = item["text"].strip()
            if len(a) < 20: continue
            key = q.lower()[:80]
            if key in seen: continue
            seen.add(key)
            pairs.append(make_pair(q, a))
            count += 1
            if count >= max_pairs: break
        print(f"  Done — {count:,} new pairs")
    except Exception as e:
        print(f"  FAILED: {e}")
    return pairs, len(ds) if 'ds' in dir() else 0

def dl_coding(seen, max_pairs, resume_index=0):
    """Python code instructions"""
    print(f"\n[2/4] Coding dataset — resume_index={resume_index}")
    pairs = []; count = 0
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        for idx, item in enumerate(ds):
            if idx < resume_index: continue
            q = item.get("instruction", "").strip()
            a = item.get("output", "").strip()
            if not q or len(a) < 20: continue
            key = q.lower()[:80]
            if key in seen: continue
            seen.add(key)
            pairs.append(make_pair(q, a))
            count += 1
            if count >= max_pairs: break
        print(f"  Done — {count:,} new pairs")
    except Exception as e:
        print(f"  FAILED: {e}")
    return pairs, len(ds) if 'ds' in dir() else 0

def dl_explanation(seen, max_pairs, resume_index=0):
    """DuoRC explanations"""
    print(f"\n[3/4] Explanation dataset — resume_index={resume_index}")
    pairs = []; count = 0
    try:
        ds = load_dataset("duorc", "SelfRC", split="train")
        for idx, item in enumerate(ds):
            if idx < resume_index: continue
            q = item.get("question", "").strip()
            answers = item.get("answers", [])
            if not answers: continue
            a = max(answers, key=len).strip()
            if not q or len(a) < 20: continue
            key = q.lower()[:80]
            if key in seen: continue
            seen.add(key)
            pairs.append(make_pair(q, a))
            count += 1
            if count >= max_pairs: break
        print(f"  Done — {count:,} new pairs")
    except Exception as e:
        print(f"  FAILED: {e}")
    return pairs, len(ds) if 'ds' in dir() else 0

def dl_conversation(seen, max_pairs, resume_index=0):
    """Alpaca general instructions"""
    print(f"\n[4/4] Alpaca (general assistant) — resume_index={resume_index}")
    pairs = []; count = 0
    try:
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        for idx, item in enumerate(ds):
            if idx < resume_index: continue
            q = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            a   = item.get("output", "").strip()
            if inp: q = f"{q}\n{inp}"
            if not q or len(a) < 20: continue
            key = q.lower()[:80]
            if key in seen: continue
            seen.add(key)
            pairs.append(make_pair(q, a))
            count += 1
            if count >= max_pairs: break
        print(f"  Done — {count:,} new pairs")
    except Exception as e:
        print(f"  FAILED: {e}")
    return pairs, len(ds) if 'ds' in dir() else 0

DOWNLOADERS = {
    "assistant":    dl_assistant,
    "coding":       dl_coding,
    "explanation":  dl_explanation,
    "conversation": dl_conversation,
}

DATASET_ORDER = ["assistant", "coding", "explanation", "conversation"]

# ═══════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════

def main():
    global _gz_has_content
    print("=" * 55)
    print("  MyAI Data Downloader")
    print(f"  Chunk size: {CHUNK_CHARS/1024/1024:.0f} MB (flush + compress after this)")
    print("=" * 55)

    # ── Step 1: Check / validate existing .gz ─────────
    prog = load_progress()
    completed = set(prog.get("completed", []))
    current_ds = prog.get("current", None)
    current_idx = prog.get("current_index", 0)

    if os.path.exists(DATA_GZ):
        ok, existing_pairs, seen, existing_chars = validate_gz(DATA_GZ)
        if not ok:
            print("\n⚠ Existing .gz is CORRUPTED.")
            choice = input("  Delete and start fresh? [y/n]: ").strip().lower()
            if choice == "y":
                os.remove(DATA_GZ)
                clear_progress()
                existing_pairs, seen, existing_chars = 0, set(), 0
                completed, current_ds, current_idx = set(), None, 0
                print("  Deleted. Starting fresh.")
            else:
                print("  Aborted. Fix or delete training_data.txt.gz manually.")
                return
        else:
            _gz_has_content = existing_chars > 0

        if existing_pairs > 0:
            print(f"\nResuming — found {existing_pairs:,} existing pairs")
            if completed:
                print(f"  Completed datasets: {', '.join(sorted(completed))}")
            if current_ds:
                print(f"  Partial: {current_ds} at index {current_idx:,}")
    else:
        seen = set()
        existing_pairs = 0
        _gz_has_content = False
        print("\nNo existing data — starting fresh")

    # ── Step 2: Download each dataset ─────────────────
    buffer      = []        # pending pairs not yet flushed
    buffer_chars = 0
    total_new   = 0

    for ds_name in DATASET_ORDER:
        if not DATASETS_TO_DOWNLOAD.get(ds_name): continue
        if ds_name in completed:
            print(f"\n[skip] {ds_name} — already completed")
            continue

        resume_idx = current_idx if ds_name == current_ds else 0
        prog["current"]       = ds_name
        prog["current_index"] = resume_idx
        save_progress(prog)

        t_ds = time.time()
        fn    = DOWNLOADERS[ds_name]
        new_pairs, ds_len = fn(seen, SIZES.get(ds_name, 999999), resume_idx)

        total_new += len(new_pairs)
        buffer    += new_pairs
        buffer_chars += sum(len(p) for p in new_pairs)

        # Update progress with end-of-dataset index so we skip it fully on resume
        prog["current_index"] = ds_len
        save_progress(prog)

        # ── Flush if buffer is large enough ──
        if buffer_chars >= CHUNK_CHARS:
            flush_buffer(buffer)
            buffer = []; buffer_chars = 0

        # Mark dataset as done
        completed.add(ds_name)
        prog["completed"]     = list(completed)
        prog["current"]       = None
        prog["current_index"] = 0
        save_progress(prog)
        print(f"  {ds_name} in {time.time()-t_ds:.0f}s")

    # ── Step 3: Final flush of remaining buffer ────────
    if buffer:
        flush_buffer(buffer)

    # ── Step 4: Summary ───────────────────────────────
    print("\n" + "=" * 55)
    try:
        _, final_pairs, _, final_chars = validate_gz(DATA_GZ)
        gz_size = os.path.getsize(DATA_GZ)
        print(f"Total pairs : {final_pairs:,}")
        print(f"Total chars : {final_chars/1024/1024:.0f} MB raw")
        print(f"File size   : {gz_size/1024/1024:.0f} MB compressed")
        print(f"New added   : {total_new:,}")
    except Exception as e:
        print(f"Summary error: {e}")

    clear_progress()

    if os.path.exists("./hf_cache"):
        shutil.rmtree("./hf_cache", ignore_errors=True)
        print("Cleared HuggingFace cache")

    print("\nDone! Run: python train.py")

if __name__ == "__main__":
    main()
