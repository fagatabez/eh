# train.py
# ── Auto-install missing packages ──────────────────────────────────────────
import sys, subprocess, os

def _ensure(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        print(f"[setup] installing {pkg}...")
        for flags in [["--break-system-packages", "-q"], ["-q"]]:
            r = subprocess.run([sys.executable, "-m", "pip", "install", pkg] + flags,
                               capture_output=True, text=True)
            if r.returncode == 0: break
        else:
            print(f"[setup] FAILED: {pkg}\n{r.stderr}"); sys.exit(1)
        print(f"[setup] {pkg} OK — restarting..."); os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure("torch"); _ensure("requests"); _ensure("numpy")
# ──────────────────────────────────────────────────────────────────────────

import gzip, json, time, signal, hashlib, threading
import torch, requests
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler
from model import MyAI, Config
from tokenizer import Tokenizer
from data import TextDataset

# ── Command-line flags ────────────────────────────────────────────────────────
# --extend   : skip data-changed reset, continue training on expanded dataset
EXTEND_MODE = "--extend" in sys.argv

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

EPOCHS        = 30
START_EPOCH   = 0
BATCH_SIZE    = 500
LEARNING_RATE = 3e-4
VOCAB_SIZE    = 60000
SAVE_EVERY    = 2
RESUME_FROM   = None
MODEL_SIZE    = "small"
VRAM_RESERVE_GB = 1.5

SERVER_URL  = "https://eh-production.up.railway.app"
SECRET_KEY  = "Dsadasdsefgtgtlubiemlodydsadasdseflubiemlody1bekekejroliwer2011elo%5dfdsfdsk"
SYNC_EVERY  = 5

MAX_SERVER_FAILURES = 3

# Worker gradient blending
GRAD_PULL_EVERY   = 10
WORKER_GRAD_BLEND = 0.3

# Push real weights to server every N epochs so workers can download them
PUSH_WEIGHTS_EVERY = 1

# Mid-epoch checkpoint
MID_EPOCH_CKPT  = "myai_midepoch.pt"
MID_EPOCH_EVERY = 200

DATA_HASH_FILE        = ".data_hash"
BEST_LOSS_FILE        = ".best_loss"
BUDGET_FILE           = ".training_budget.json"
TRAINING_COMPLETE_FILE = ".training_complete.json"
AUTOSCALE_FILE        = ".autoscale.json"

# ── Progressive training ──────────────────────────
# When loss drops below TARGET_LOSS, automatically expand
# the training slice by PROGRESSIVE_FACTOR (up to the budget max).
PROGRESSIVE         = True
TARGET_LOSS         = 0.50
PROGRESSIVE_FACTOR  = 2.0    # double the slice each time
# ─────────────────────────────────────────────────

# ── Phase training ────────────────────────────────
# ALWAYS enabled. Trains in 100k-char cumulative chunks until loss ≤ 0.50,
# then expands to ALL chars seen so far before adding the next 100k.
#
# Phase 1: trains on 100k chars → must reach loss ≤ 0.50
# Phase 2: trains on 200k chars (all of phase 1 + 100k new) → loss ≤ 0.50
# Phase 3: trains on 300k chars → loss ≤ 0.50  ...and so on.
#
# After hitting the target, a "generalization pass" runs on rephrased +
# out-of-window data to prevent pure memorization.
PHASE_TRAINING        = True
PHASE_SIZE            = 100_000    # new chars added per phase
PHASE_FILE            = ".phase_state.json"

# Loss each phase must reach before expanding.
TARGET_LOSS_PER_PHASE = 0.50

KAGGLE_PHASE_SIZE     = 100_000

# ═══════════════════════════════════════════════════
#   TRAINING BUDGET — how much data to use
# ═══════════════════════════════════════════════════

def is_kaggle():
    """Detect whether we're running inside a Kaggle notebook/kernel."""
    return (
        os.path.exists("/kaggle") or
        os.path.exists("/kaggle/working") or
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ or
        "KAGGLE_DATA_PROXY_TOKEN" in os.environ
    )

def parse_size(s):
    """
    Parse a human-readable size string → int (number of chars).
    Accepts:  1m  1.5m  500k  1.23b  1230000  1230k  etc.
    Case-insensitive.
    """
    s = s.strip().lower().replace(",", "").replace("_", "")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000,
                   "t": 1_000_000_000_000}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                return int(float(s[:-1]) * mult)
            except ValueError:
                break
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"Cannot parse size: '{s}' — use formats like: 1m  500k  1.5b  1000000")

def format_size(n):
    """Int → human readable string.  1_500_000 → '1.5m'"""
    if n >= 1_000_000_000: return f"{n/1_000_000_000:.2g}b"
    if n >= 1_000_000:     return f"{n/1_000_000:.2g}m"
    if n >= 1_000:         return f"{n/1_000:.2g}k"
    return str(n)

def load_budget():
    """Load saved budget from disk.  Returns dict or None."""
    if os.path.exists(BUDGET_FILE):
        try:
            with open(BUDGET_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_budget(b):
    with open(BUDGET_FILE, "w") as f:
        json.dump(b, f, indent=2)

def get_or_ask_budget(total_chars):
    """
    Returns the number of chars to use for THIS run.
    On Kaggle → max (total_chars).
    On first run → prompts user.
    On subsequent runs → loads from file.
    In extend mode / after training complete → asks "how much more?"
    Handles the user/kaggle split in the budget file.
    """
    on_kaggle = is_kaggle()

    budget = load_budget()
    complete = load_training_complete()

    if on_kaggle:
        # On Kaggle with no budget file → auto-set 1m phase target.
        # Phase training will then expand by 1m each time loss ≤ 0.50.
        # This means no prompts ever appear on Kaggle — fully automatic.
        if budget is None:
            kaggle_start = min(1_000_000, total_chars)
            budget = {"kaggle": kaggle_start, "current": kaggle_start, "user": kaggle_start}
            save_budget(budget)
            print(f"\n[Kaggle] No budget file found — auto-set to {format_size(kaggle_start)} "
                  f"(phases of {format_size(KAGGLE_PHASE_SIZE)}, expanding at loss ≤ {TARGET_LOSS_PER_PHASE})")
        current = budget.get("current", budget.get("kaggle", total_chars))
        current = min(current, total_chars)
        print(f"\n[Kaggle] Training budget: {format_size(current)} / {format_size(total_chars)} chars")
        budget["kaggle"]  = current
        budget["current"] = current
        save_budget(budget)
        return current

    # ── Extend mode: training_complete.json exists → ask "how much more?" ──────
    if (EXTEND_MODE or complete) and complete:
        trained_chars = complete.get("chars_trained", 0)
        prev_loss     = complete.get("best_loss", float("inf"))
        prev_total    = complete.get("total_chars", total_chars)

        # Check autoscale config
        asc = get_autoscale()
        if asc.get("enabled"):
            step = asc["step_chars"]
            new_chars = min(trained_chars + step, total_chars)
            print(f"\n[Auto-expand] Previously trained: {format_size(trained_chars)} / {format_size(total_chars)} "
                  f"chars (loss={prev_loss:.4f})")
            print(f"  Auto-expanding by {format_size(step)} → {format_size(new_chars)} chars")
        else:
            print(f"\nPreviously trained: {format_size(trained_chars)} / {format_size(total_chars)} chars  "
                  f"(best loss: {prev_loss:.4f})")
            print(f"How much MORE to train? (added on top of already-trained data)")
            print(f"  Examples: 200k  1m  5m  all  (all = train everything)")
            print(f"  Remaining: {format_size(total_chars - trained_chars)} chars untrained")
            while True:
                raw = input("  Add chars [all]: ").strip()
                if raw == "" or raw.lower() == "all":
                    new_chars = total_chars
                    break
                try:
                    add = parse_size(raw)
                    new_chars = min(trained_chars + add, total_chars)
                    if new_chars <= 0:
                        print("  Must be > 0"); continue
                    break
                except ValueError as e:
                    print(f"  {e}")

        budget = {"user": new_chars, "kaggle": 0, "current": new_chars}
        save_budget(budget)
        clear_training_complete()
        print(f"  Training slice: {format_size(new_chars)} chars")
        return new_chars

    # ── Local machine: already set ─────────────────────────────────────────────
    if budget and "user" in budget:
        current = budget.get("current", budget["user"])
        current = min(current, total_chars)
        user_max = min(budget["user"], total_chars)
        print(f"\nTraining budget: {format_size(current)} / {format_size(user_max)} chars "
              f"(full data: {format_size(total_chars)})")
        print(f"  [budget file: {BUDGET_FILE}]  Delete it to change the budget.")
        return current

    # ── First run: ask ─────────────────────────────────────────────────────────
    # Also ask about auto-scale here (once, stored permanently)
    get_autoscale()   # sets up autoscale config if not already done

    print(f"\nTraining data: {format_size(total_chars)} chars available.")
    print("How many chars do you want to train on?")
    print("  Examples: 1m  500k  1.5b  all  (or just press Enter for all)")
    print("  Tip: start small (1m–10m) on CPU, use more on Kaggle GPU.")
    while True:
        raw = input("  Budget [all]: ").strip()
        if raw == "" or raw.lower() == "all":
            user_chars = total_chars
            break
        try:
            user_chars = parse_size(raw)
            user_chars = min(user_chars, total_chars)
            if user_chars <= 0:
                print("  Must be > 0"); continue
            break
        except ValueError as e:
            print(f"  {e}")

    budget = {
        "user":    user_chars,
        "kaggle":  0,
        "current": user_chars,
    }
    save_budget(budget)
    print(f"  Saved to {BUDGET_FILE}  (delete to change later)")
    return user_chars

def update_budget_current(new_chars):
    """After progressive expansion, record the new current slice size."""
    budget = load_budget()
    if budget is None: budget = {}
    budget["current"] = new_chars
    save_budget(budget)

def get_budget_max(total_chars):
    """Return the cap: user budget (local) or total_chars (Kaggle)."""
    budget = load_budget()
    if budget is None: return total_chars
    if is_kaggle(): return total_chars
    return min(budget.get("user", total_chars), total_chars)

SYSTEM_PROMPT = """
You are MyAI. You are helpful, honest, and curious.
You never make things up. If you don't know, say so.
You can write Python code. You speak simply and clearly.
"""

SIZE_CONFIGS = {
    "tiny":   dict(embed_dim=128, num_heads=4, num_layers=4),
    "small":  dict(embed_dim=256, num_heads=8, num_layers=6),
    "medium": dict(embed_dim=512, num_heads=8, num_layers=8),
}

stop_training     = False
server_fail_count = 0

def handle_interrupt(sig, frame):
    global stop_training
    print("\nStopping — saving checkpoint..."); stop_training = True

signal.signal(signal.SIGINT,  handle_interrupt)
signal.signal(signal.SIGTERM, handle_interrupt)

def fmt(s):
    if s < 60:   return f"{int(s)}s"
    if s < 3600: return f"{int(s//60)}m {int(s%60)}s"
    return f"{int(s//3600)}h {int((s%3600)//60)}m"

# ═══════════════════════════════════════════════════
#   BACKGROUND FILE SYNC WATCHER
#   Watches local files for changes and instantly pushes
#   them to the server when modified — no manual upload.
#   Files watched: tokenizer.json, myai.pt
# ═══════════════════════════════════════════════════

_sync_mtimes  = {}   # path → last mtime we synced
_sync_alive   = False

def _get_mtime(path):
    try:    return os.path.getmtime(path)
    except: return 0

def _push_file(path):
    """Push a single file to server immediately."""
    hdrs = {"X-Secret-Key": SECRET_KEY}
    try:
        if path.endswith("tokenizer.json"):
            with open(path) as f:
                r = requests.post(f"{SERVER_URL}/tokenizer",
                    json=json.load(f), headers=hdrs, timeout=30)
            ok = r and r.status_code == 200
        elif path.endswith(".pt"):
            with open(path, "rb") as f:
                r = requests.post(f"{SERVER_URL}/model",
                    data=f.read(), headers=hdrs, timeout=90)
            ok = r and r.status_code == 200
        else:
            return
        if ok:
            print(f"[sync] ✓ {os.path.basename(path)} pushed to server")
        else:
            print(f"[sync] ✗ {os.path.basename(path)} push failed (HTTP {r.status_code if r else '?'})")
    except Exception as e:
        print(f"[sync] ✗ {os.path.basename(path)}: {e}")

def _file_watcher():
    """Background thread: push files to server whenever they change on disk."""
    WATCH = ["tokenizer.json", "myai.pt"]
    while _sync_alive:
        time.sleep(3)
        if not SERVER_URL: break
        for fname in WATCH:
            mtime = _get_mtime(fname)
            if mtime == 0: continue             # file doesn't exist yet
            if mtime != _sync_mtimes.get(fname):
                _sync_mtimes[fname] = mtime     # record so we don't push twice
                threading.Thread(target=_push_file, args=(fname,), daemon=True).start()

def start_sync_watcher():
    global _sync_alive
    if not SERVER_URL: return
    _sync_alive = True
    # Record current mtimes so we don't push at startup unnecessarily
    for fname in ["tokenizer.json", "myai.pt"]:
        _sync_mtimes[fname] = _get_mtime(fname)
    threading.Thread(target=_file_watcher, daemon=True).start()
    print("[sync] File watcher started — tokenizer.json and myai.pt auto-sync on change")

def stop_sync_watcher():
    global _sync_alive
    _sync_alive = False

def force_push_now(path):
    """Immediately push a file, also update the mtime tracker."""
    if not SERVER_URL: return
    _push_file(path)
    _sync_mtimes[path] = _get_mtime(path)

# ═══════════════════════════════════════════════════
#   SERVER HELPERS (with retry limit)
# ═══════════════════════════════════════════════════

def server_get(path, **kw):
    global SERVER_URL, server_fail_count
    if not SERVER_URL: return None
    try:
        r = requests.get(f"{SERVER_URL}{path}", **kw)
        server_fail_count = 0; return r
    except Exception as e:
        server_fail_count += 1
        if server_fail_count >= MAX_SERVER_FAILURES:
            print(f"[net] {MAX_SERVER_FAILURES} failures — disabling server for this run.")
            SERVER_URL = ""
        else:
            print(f"[net] {e} ({server_fail_count}/{MAX_SERVER_FAILURES})")
        return None

def server_post(path, **kw):
    global SERVER_URL, server_fail_count
    if not SERVER_URL: return None
    try:
        r = requests.post(f"{SERVER_URL}{path}", **kw)
        server_fail_count = 0; return r
    except Exception as e:
        server_fail_count += 1
        if server_fail_count >= MAX_SERVER_FAILURES:
            print(f"[net] {MAX_SERVER_FAILURES} failures — disabling server for this run.")
            SERVER_URL = ""
        else:
            print(f"[net] {e} ({server_fail_count}/{MAX_SERVER_FAILURES})")
        return None

# ═══════════════════════════════════════════════════
#   DATA HELPERS
# ═══════════════════════════════════════════════════

def find_data_path():
    for p in ["training_data.txt.gz", "training_data.txt"]:
        if os.path.exists(p): return p
    return None

def compute_data_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f: h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()

def check_data_changed():
    path = find_data_path()
    if path is None:
        raise FileNotFoundError(
            "No training data!\nUpload training_data.txt.gz or .txt, or run download_data.py")
    print(f"Data file: {path}")
    cur = compute_data_hash(path)
    if not os.path.exists(DATA_HASH_FILE):
        open(DATA_HASH_FILE, "w").write(cur)
        print(f"Hash stored ({cur[:8]}) — first run"); return True
    stored = open(DATA_HASH_FILE).read().strip()
    if cur == stored:
        print(f"Data unchanged ({cur[:8]})"); return False
    open(DATA_HASH_FILE, "w").write(cur)
    if EXTEND_MODE:
        print(f"Data changed ({stored[:8]}→{cur[:8]}) — EXTEND MODE: keeping model, expanding data")
        return False   # treat as "not changed" so we keep the checkpoint
    print(f"Data changed ({stored[:8]}→{cur[:8]}) — fresh training"); return True

def load_data():
    path = find_data_path()
    if path is None: raise FileNotFoundError("No training data!")
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f: return f.read()
    with open(path, encoding="utf-8") as f: return f.read()

# ═══════════════════════════════════════════════════
#   OVERWRITE PROTECTION
# ═══════════════════════════════════════════════════

def load_best_loss():
    try: return float(open(BEST_LOSS_FILE).read().strip())
    except: return float("inf")

def save_best_loss(loss):
    try: open(BEST_LOSS_FILE, "w").write(str(loss))
    except: pass

# ═══════════════════════════════════════════════════
#   PHASE TRAINING
#   Splits total budget into equal phases of PHASE_SIZE chars.
#   Each phase trains fully then saves a checkpoint.
#   Allows short sessions to make real progress.
#
#   phase_state.json tracks:
#     phase_size    : chars per phase
#     total_target  : total chars to train (full budget)
#     current_phase : which phase we're on (0-indexed)
#     total_phases  : how many phases total
#     chars_done    : chars trained so far across all phases
# ═══════════════════════════════════════════════════

def load_phase_state():
    try:
        with open(PHASE_FILE) as f:
            return json.load(f)
    except: return None

def save_phase_state(state):
    with open(PHASE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def clear_phase_state():
    try:
        if os.path.exists(PHASE_FILE): os.remove(PHASE_FILE)
    except: pass

def _build_phase_state(size, total_budget_chars):
    """Create a fresh phase state dict for the given phase size."""
    total_phases = max(1, -(-total_budget_chars // size))  # ceiling division
    return {
        "phase_size":    size,
        "total_target":  total_budget_chars,
        "current_phase": 0,
        "total_phases":  total_phases,
        "chars_done":    0,
        "target_loss":   TARGET_LOSS_PER_PHASE,
        "enabled":       True,
    }

def ask_phase_config(total_budget_chars):
    """Auto-configure phases — never prompts. Always 100k cumulative.
    On Kaggle uses KAGGLE_PHASE_SIZE (same 100k default)."""
    size = KAGGLE_PHASE_SIZE if is_kaggle() else PHASE_SIZE
    size = min(size, total_budget_chars)
    total_phases = max(1, -(-total_budget_chars // size))
    state = {
        "phase_size":    size,
        "total_target":  total_budget_chars,
        "current_phase": 0,
        "total_phases":  total_phases,
        "chars_done":    0,
        "target_loss":   TARGET_LOSS_PER_PHASE,
        "enabled":       True,
        "cumulative":    True,
    }
    save_phase_state(state)
    print(f"\n[Phase training] AUTO-ENABLED (cumulative expansion):")
    print(f"  Chunk size : +{format_size(size)} chars each phase")
    print(f"  Phases     : {total_phases}  (window grows: {format_size(size)} → {format_size(size*2)} → ...)")
    print(f"  Target loss: ≤ {TARGET_LOSS_PER_PHASE} before expanding")
    print(f"  Total      : {format_size(total_budget_chars)} chars")
    return state

def get_phase_config(total_budget_chars):
    """Always returns a phase config — auto-creates if missing, never disables."""
    state = load_phase_state()
    if state is not None:
        if not state.get("enabled", True):
            clear_phase_state()
            return ask_phase_config(total_budget_chars)
        if state.get("total_target", 0) != total_budget_chars:
            print(f"  [phase] Budget changed — resetting phase state")
            clear_phase_state()
            return ask_phase_config(total_budget_chars)
        # Patch old saved states missing new fields
        changed = False
        if "target_loss" not in state:
            state["target_loss"] = TARGET_LOSS_PER_PHASE; changed = True
        if "cumulative" not in state:
            state["cumulative"] = True; changed = True
        if changed:
            save_phase_state(state)
        ph = state["current_phase"]
        ph_total = state["total_phases"]
        ph_done  = state["chars_done"]
        print(f"\n  Resuming phase {ph+1}/{ph_total} "
              f"({format_size(ph_done)}/{format_size(total_budget_chars)} trained so far)")
        return state
    return ask_phase_config(total_budget_chars)

# ═══════════════════════════════════════════════════
#   TRAINING COMPLETE / EXTEND MODE
#   Saved when training finishes naturally (not Ctrl+C).
#   On next run, allows you to "extend" training on
#   more data without losing the trained model.
# ═══════════════════════════════════════════════════

def save_training_complete(chars_trained, total_chars, best_loss, data_hash):
    info = {
        "chars_trained": chars_trained,
        "total_chars":   total_chars,
        "best_loss":     round(best_loss, 6),
        "data_hash":     data_hash,
        "finished_at":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(TRAINING_COMPLETE_FILE, "w") as f:
        json.dump(info, f, indent=2)

def upload_state_files():
    """
    Upload all state/config files to the server after training ends.
    Uploads both the dot version (e.g. .best_loss) AND the plain version
    (best_loss) if they exist — download_results.py will grab whichever exist.

    Also uploads the latest myai_epochN.pt checkpoint.
    """
    if not SERVER_URL:
        return

    print("\n[sync] Uploading state files to server for download_results.py ...")
    hdrs = {"X-Secret-Key": SECRET_KEY}

    # Pairs: (filename_on_disk, name_to_send_to_server)
    # We send the file under its actual name — server stores it under that name.
    candidates = [
        # dot versions (hidden files)
        ".best_loss",
        ".data_hash",
        ".training_budget.json",
        ".phase_state.json",
        ".training_complete.json",
        ".autoscale.json",
        # plain versions (some tools write without dot)
        "best_loss",
        "data_hash",
        "training_budget.json",
        "phase_state.json",
        "training_complete.json",
        "autoscale.json",
    ]

    # Also find the latest myai_epochN.pt
    import glob as _glob, re as _re
    epoch_ckpts = sorted(
        [f for f in _glob.glob("myai_epoch*.pt")
         if _re.match(r'^myai_epoch\d+\.pt$', f)],
        key=lambda f: int(_re.search(r'\d+', f).group())
    )
    if epoch_ckpts:
        candidates.append(epoch_ckpts[-1])   # only the latest

    uploaded = []; missing = []
    for fname in candidates:
        if not os.path.exists(fname):
            missing.append(fname)
            continue
        try:
            with open(fname, "rb") as f:
                data = f.read()
            r = requests.post(
                f"{SERVER_URL}/file/{fname}",
                data=data, headers=hdrs, timeout=60
            )
            if r and r.status_code == 200:
                size = len(data)
                sz_str = f"{size/1024/1024:.1f}MB" if size > 1024*1024 else f"{size/1024:.0f}KB"
                print(f"  ✓  {fname}  ({sz_str})")
                uploaded.append(fname)
            else:
                print(f"  ✗  {fname}  (HTTP {r.status_code if r else '?'})")
        except Exception as e:
            print(f"  ✗  {fname}  ({e})")

    if uploaded:
        print(f"[sync] State files uploaded: {len(uploaded)} file(s)")
        print(f"[sync] Run  python download_results.py  on your PC to get everything.")

def load_training_complete():
    try:
        with open(TRAINING_COMPLETE_FILE) as f:
            return json.load(f)
    except: return None

def clear_training_complete():
    try:
        if os.path.exists(TRAINING_COMPLETE_FILE):
            os.remove(TRAINING_COMPLETE_FILE)
    except: pass

# ═══════════════════════════════════════════════════
#   AUTO-SCALE CONFIG
#   Asks once: "auto-expand after each run? by how much?"
#   Saved to .autoscale.json
# ═══════════════════════════════════════════════════

def load_autoscale():
    try:
        with open(AUTOSCALE_FILE) as f:
            return json.load(f)
    except: return None

def save_autoscale(cfg):
    with open(AUTOSCALE_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

def ask_autoscale():
    """Ask the user once about auto-scaling. On Kaggle, auto-enables silently."""
    # On Kaggle there's no interactive terminal — auto-enable with 1m steps
    if is_kaggle():
        cfg = {"enabled": True, "step_chars": 1_000_000}
        save_autoscale(cfg)
        print(f"  [Kaggle] Auto-expand enabled: +1m chars per run (saved to {AUTOSCALE_FILE})")
        return cfg

    print("\nAuto-expand after training finishes?")
    print("  If yes, next run will automatically train on extra chars.")
    raw = input("  Auto-expand? [yes/no]: ").strip().lower()
    if raw not in ("yes", "y"):
        cfg = {"enabled": False}
        save_autoscale(cfg)
        print("  Auto-expand disabled.")
        return cfg

    print("  How many chars to add after each run?")
    print("  Examples: 200k  1m  5m  500k")
    while True:
        raw2 = input("  Add chars: ").strip()
        try:
            step = parse_size(raw2)
            if step <= 0:
                print("  Must be > 0"); continue
            break
        except ValueError as e:
            print(f"  {e}")

    cfg = {"enabled": True, "step_chars": step}
    save_autoscale(cfg)
    print(f"  Auto-expand: +{format_size(step)} per run — saved to {AUTOSCALE_FILE}")
    print(f"  Delete {AUTOSCALE_FILE} to change this setting.")
    return cfg

def get_autoscale():
    """Return autoscale config, asking user if not set yet."""
    cfg = load_autoscale()
    if cfg is None:
        cfg = ask_autoscale()
    return cfg

def checkpoint_loss(path):
    try:
        ckpt = torch.load(path, map_location="cpu")
        return float(ckpt.get("best_loss", float("inf")))
    except: return float("inf")

# ═══════════════════════════════════════════════════
#   MID-EPOCH CHECKPOINT
# ═══════════════════════════════════════════════════

def save_mid_epoch(model, optimizer, epoch, step, cfg, data_hash, best_loss=float("inf")):
    m = model.module if hasattr(model, "module") else model
    torch.save({"model": m.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": step, "config": vars(cfg),
                "system": SYSTEM_PROMPT, "data_hash": data_hash,
                "best_loss": best_loss, "mid_epoch": True}, MID_EPOCH_CKPT)

def load_mid_epoch(data_hash):
    if not os.path.exists(MID_EPOCH_CKPT): return None
    try:
        c = torch.load(MID_EPOCH_CKPT, map_location="cpu")
        if not c.get("mid_epoch"): return None
        if c.get("data_hash") != data_hash:
            print("[resume] Mid-epoch checkpoint is for different data — ignoring"); return None
        return c
    except: return None

def delete_mid_epoch():
    try:
        if os.path.exists(MID_EPOCH_CKPT): os.remove(MID_EPOCH_CKPT)
    except: pass

# ═══════════════════════════════════════════════════
#   CHECKPOINT PRIORITY PICKER
# ═══════════════════════════════════════════════════

def pick_resume_checkpoint(data_changed, data_hash):
    if RESUME_FROM is not None:
        print(f"Manual resume: {RESUME_FROM}"); return RESUME_FROM, False

    if data_changed:
        print("Data changed — starting fresh"); return None, False

    # 1. Mid-epoch (exact step, highest priority)
    mid = load_mid_epoch(data_hash)
    if mid:
        e, s = mid.get("epoch", 0), mid.get("step", 0)
        print(f"✓ Mid-epoch checkpoint — resuming epoch {e+1}, step {s}")
        return MID_EPOCH_CKPT, True

    # 2. Latest myai_epochN.pt
    cks = sorted(
        [f for f in os.listdir(".") if f.startswith("myai_epoch") and f.endswith(".pt")],
        key=lambda f: int(f.replace("myai_epoch","").replace(".pt",""))
    )
    if cks:
        print(f"✓ Resuming from {cks[-1]}"); return cks[-1], False

    # 3. myai.pt
    if os.path.exists("myai.pt"):
        print("✓ Resuming from myai.pt"); return "myai.pt", False

    # 4. Server download (Kaggle session reset)
    print("No local checkpoint — checking server...")
    r = server_get("/model", timeout=60)
    if r and r.status_code == 200 and len(r.content) > 1000:
        path = "_server_model.pt"
        with open(path, "wb") as f: f.write(r.content)
        print(f"✓ Downloaded from server ({len(r.content)/1024/1024:.1f} MB)")
        return path, False

    print("Starting fresh"); return None, False

# ═══════════════════════════════════════════════════
#   MODEL SAVE (with overwrite protection)
# ═══════════════════════════════════════════════════

def save_model(model, optimizer, epoch, cfg, path="myai.pt", step=0, best_loss=float("inf"), trained_chars=0):
    # Overwrite protection for the main myai.pt only
    if path == "myai.pt" and os.path.exists(path):
        disk_best = min(checkpoint_loss(path), load_best_loss())
        if best_loss > disk_best + 0.01:
            print(f"  ⚠ Not saving — current loss ({best_loss:.4f}) worse than "
                  f"saved ({disk_best:.4f}). Model protected.")
            return False

    m = model.module if hasattr(model, "module") else model
    torch.save({"model": m.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "step": step, "config": vars(cfg),
                "system": SYSTEM_PROMPT, "best_loss": best_loss,
                "trained_chars": trained_chars}, path)
    if best_loss < load_best_loss(): save_best_loss(best_loss)
    print(f"Saved: {path}  (loss={best_loss:.4f})")
    return True

# ═══════════════════════════════════════════════════
#   CHECKPOINT RESUME HELPER
# ═══════════════════════════════════════════════════

def apply_ckpt(ckpt, model, optimizer, cfg, device, interactive=False):
    ckpt_vocab = ckpt.get("config", {}).get("vocab_size", 0)
    if ckpt_vocab and ckpt_vocab != cfg.vocab_size:
        if interactive:
            c = input(f"Vocab mismatch ckpt={ckpt_vocab} tok={cfg.vocab_size}. [keep/fresh]: ").strip().lower()
            if c == "fresh": return 0, 0
        cfg.vocab_size = ckpt_vocab
    m = model.module if hasattr(model, "module") else model
    try: m.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        print(f"  Weight load failed ({e}) — fresh weights"); return 0, 0
    if "optimizer" in ckpt:
        try: optimizer.load_state_dict(ckpt["optimizer"])
        except: pass
    return ckpt.get("epoch", 0), ckpt.get("step", 0)

# ═══════════════════════════════════════════════════
#   SERVER STATS
# ═══════════════════════════════════════════════════

def _push_stats_worker(epoch, total_epochs, loss, lr, elapsed, eta,
                       message=None, step=0, total_steps=0):
    """Actual network call — always runs in a background thread."""
    server_post("/training_update",
        headers={"X-Secret-Key": SECRET_KEY},
        json={"epoch": epoch, "total_epochs": total_epochs,
              "step": step, "total_steps": total_steps,
              "loss": round(float(loss), 4), "lr": round(float(lr), 6),
              "status": "training", "elapsed": int(elapsed),
              "eta": int(eta), "message": message}, timeout=5)

def push_stats(epoch, total_epochs, loss, lr, elapsed, eta,
               message=None, step=0, total_steps=0):
    """Fire-and-forget — never blocks the training loop."""
    threading.Thread(
        target=_push_stats_worker,
        args=(epoch, total_epochs, loss, lr, elapsed, eta),
        kwargs={"message": message, "step": step, "total_steps": total_steps},
        daemon=True
    ).start()

# ═══════════════════════════════════════════════════
#   DEVICE DETECTION
# ═══════════════════════════════════════════════════

def detect_device():
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"GPU detected: {n} GPU(s)")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)")
        return "cuda", n
    print(f"No GPU — CPU ({os.cpu_count()} cores)"); return "cpu", 0

def get_gpu_info():
    return [torch.cuda.get_device_properties(i).total_memory / 1024**3
            for i in range(torch.cuda.device_count())]

# ═══════════════════════════════════════════════════
#   PER-GPU BATCH PROBING
# ═══════════════════════════════════════════════════

def probe_safe_batch(model, device, vram_gb, seq_len):
    usable_gb = vram_gb - VRAM_RESERVE_GB
    if usable_gb <= 0: return 1
    model.train()
    probe = max(1, int(usable_gb * 512)); last_good = 1

    while probe >= 1:
        try:
            torch.cuda.empty_cache()
            dx = torch.zeros(probe, seq_len - 1, dtype=torch.long, device=device)
            dy = torch.zeros(probe, seq_len - 1, dtype=torch.long, device=device)
            with torch.amp.autocast("cuda"):
                logits = model(dx)
                loss   = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), dy.reshape(-1))
            loss.backward(); model.zero_grad(set_to_none=True)
            del dx, dy, logits, loss; torch.cuda.empty_cache()
            last_good = probe; break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); model.zero_grad(set_to_none=True); probe //= 2

    lo, hi = last_good, last_good * 2
    while lo < hi - 1:
        mid = (lo + hi) // 2
        try:
            torch.cuda.empty_cache()
            dx = torch.zeros(mid, seq_len - 1, dtype=torch.long, device=device)
            dy = torch.zeros(mid, seq_len - 1, dtype=torch.long, device=device)
            with torch.amp.autocast("cuda"):
                logits = model(dx)
                loss   = nn.CrossEntropyLoss()(logits.reshape(-1, logits.size(-1)), dy.reshape(-1))
            loss.backward(); model.zero_grad(set_to_none=True)
            del dx, dy, logits, loss; torch.cuda.empty_cache(); lo = mid
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); model.zero_grad(set_to_none=True); hi = mid

    result = max(1, min(lo, BATCH_SIZE // max(1, torch.cuda.device_count())))
    print(f"  GPU {device.index}: {vram_gb:.1f}GB → safe batch = {result}")
    return result

# ═══════════════════════════════════════════════════
#   WORKER GRADIENT SYSTEM
# ═══════════════════════════════════════════════════

_wg_cache = {}; _wg_loss = 0.0; _wg_count = 0
_wg_lock  = threading.Lock(); _wg_alive = False

def _wg_fetcher():
    global _wg_cache, _wg_loss, _wg_count
    while _wg_alive:
        time.sleep(5)
        try:
            r = server_get("/get_gradients", headers={"X-Secret-Key": SECRET_KEY}, timeout=10)
            if not r or r.status_code != 200: continue
            data = r.json(); losses = data.pop("losses", [])
            if not data: continue
            avg = {n: torch.tensor(g, dtype=torch.float32).mean(dim=0) for n, g in data.items()}
            with _wg_lock:
                _wg_cache = avg
                _wg_loss  = sum(losses)/len(losses) if losses else 0.0
                _wg_count = len(losses)
        except: pass

def start_wg_thread():
    global _wg_alive; _wg_alive = True
    threading.Thread(target=_wg_fetcher, daemon=True).start()

def stop_wg_thread():
    global _wg_alive; _wg_alive = False

def blend_worker_grads(model, blend=WORKER_GRAD_BLEND):
    with _wg_lock:
        if not _wg_cache: return False
        cache = _wg_cache; cnt = _wg_count
    m = model.module if hasattr(model, "module") else model
    ok = 0
    for name, param in m.named_parameters():
        if param.grad is None or name not in cache: continue
        wg = cache[name].to(param.device)
        if wg.shape == param.grad.shape:
            param.grad.mul_(1 - blend).add_(wg, alpha=blend); ok += 1
    return ok > 0

def flush_worker_grads(model, optimizer):
    with _wg_lock:
        if not _wg_cache: return 0
        cache = dict(_wg_cache); cnt, lss = _wg_count, _wg_loss
        _wg_cache.clear()
    optimizer.zero_grad()
    m = model.module if hasattr(model, "module") else model
    ok = 0
    for name, param in m.named_parameters():
        if name in cache and param.requires_grad:
            g = cache[name].to(param.device)
            if g.shape == param.data.shape:
                param.grad = g; ok += 1
    if ok:
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optimizer.step()
        print(f"  [workers] end-of-epoch flush | {cnt} worker(s) | avg loss:{lss:.4f}")
    return ok

# ═══════════════════════════════════════════════════
#   MICRO-STEP SERVER SYNC
#   Stats (tiny JSON) are pushed every 10 steps.
#   Model weights (135 MB) are NOT pushed here —
#   uploading 135 MB every 10 steps was OOM-killing
#   Railway. Weights sync end-of-phase via file watcher.
# ═══════════════════════════════════════════════════

def _push_model_microstep(model, optimizer, epoch, step, cfg, best_loss):
    """No-op — kept so call sites don't break."""
    pass


# ═══════════════════════════════════════════════════
#   CPU TRAINING
# ═══════════════════════════════════════════════════

def _make_loader_cpu(full_dataset, current_chars, total_chars, cpu_batch):
    """Slice the dataset proportionally to current_chars, return (loader, use_n)."""
    full_n = len(full_dataset)
    if current_chars >= total_chars:
        use_n = full_n
    else:
        use_n = max(1, int(full_n * current_chars / total_chars))
    ds = Subset(full_dataset, range(use_n)) if use_n < full_n else full_dataset
    return DataLoader(ds, batch_size=cpu_batch, shuffle=True), use_n

def _run_one_phase_cpu(model, optimizer, scheduler, loss_fn, loader, cfg,
                       device, data_hash, run_best_loss, actual_start,
                       start_step, t0, epoch_times, phase_label="",
                       target_loss=0.0):
    """
    Run up to EPOCHS epochs on the given loader.
    If target_loss > 0, stops early as soon as epoch avg loss ≤ target_loss.
    Returns (run_best_loss, last_epoch, stopped_early).
    """
    global stop_training
    last_epoch   = actual_start
    w_applies    = 0
    total_steps  = len(loader)
    phase_done   = False   # set True when target_loss reached

    for epoch in range(actual_start, EPOCHS):
        if stop_training or phase_done: break
        last_epoch = epoch
        ep_start   = time.time(); total_loss = 0.0; ep_steps = 0
        skip_to    = start_step if epoch == actual_start else 0
        start_step = 0

        for i, batch in enumerate(loader):
            if i < skip_to: continue
            if stop_training:
                save_mid_epoch(model, optimizer, epoch, i, cfg, data_hash, run_best_loss)
                print(f"Mid-epoch saved (epoch={epoch+1}, step={i}) — safe to restart"); break

            x = batch[:, :-1].to(device); y = batch[:, 1:].to(device)
            logits = model(x)
            loss   = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            optimizer.zero_grad(); loss.backward()

            if SERVER_URL and GRAD_PULL_EVERY > 0 and (i+1) % GRAD_PULL_EVERY == 0:
                if blend_worker_grads(model): w_applies += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); ep_steps += 1

            if MID_EPOCH_EVERY > 0 and (i+1) % MID_EPOCH_EVERY == 0:
                save_mid_epoch(model, optimizer, epoch, i+1, cfg, data_hash, run_best_loss)

            if SERVER_URL and (i+1) % 10 == 0:
                ela10 = time.time()-ep_start
                eta10 = ela10/max(i+1-skip_to,1)*(total_steps-i-1)
                msg10 = f"{phase_label}Epoch {epoch+1} | {i+1}/{total_steps} ({(i+1)/total_steps*100:.0f}%) | loss:{loss.item():.4f}"
                threading.Thread(target=push_stats,
                    args=(epoch+1, EPOCHS, loss.item(), scheduler.get_last_lr()[0],
                          time.time()-t0, eta10, msg10),
                    kwargs={"step": i+1, "total_steps": total_steps},
                    daemon=True).start()
                threading.Thread(
                    target=lambda m=model, o=optimizer, e=epoch, s=i+1, c=cfg, bl=run_best_loss:
                        _push_model_microstep(m, o, e, s, c, bl),
                    daemon=True).start()

            if (i+1) % 50 == 0:
                pct = (i+1)/total_steps*100; ela = time.time()-ep_start
                eta = ela/max(i+1-skip_to,1)*(total_steps-i-1)
                wi  = f" workers:{w_applies}" if w_applies else ""
                tgt = f" target:≤{target_loss}" if target_loss > 0 else ""
                print(f"{phase_label}Epoch {epoch+1} | {i+1}/{total_steps} ({pct:.0f}%) | loss:{loss.item():.4f}{wi}{tgt} | eta:{fmt(eta)}")

        if stop_training: break

        ep_time = time.time()-ep_start; avg = total_loss/max(ep_steps,1)
        eta_tot = (sum(epoch_times+[ep_time])/(len(epoch_times)+1))*(EPOCHS-epoch-1)
        epoch_times.append(ep_time)
        optimizer.step(); scheduler.step()

        if w_applies: print(f"  [workers] blended {w_applies} times this epoch")
        w_applies = 0
        if SERVER_URL: flush_worker_grads(model, optimizer)
        if avg < run_best_loss: run_best_loss = avg

        # ── Check if this phase's target loss is reached ──
        if target_loss > 0 and avg <= target_loss:
            print(f"  ✓ {phase_label}Target loss reached ({avg:.4f} ≤ {target_loss}) — phase complete!")
            phase_done = True

        msg = f"{phase_label}Epoch {epoch+1}/{EPOCHS} loss:{avg:.4f} took:{fmt(ep_time)} eta:{fmt(eta_tot)}"
        print(msg)
        push_stats(epoch+1, EPOCHS, avg, scheduler.get_last_lr()[0],
                   time.time()-t0, eta_tot, msg, step=total_steps, total_steps=total_steps)

        if (epoch+1) % SAVE_EVERY == 0:
            save_model(model, optimizer, epoch+1, cfg, f"myai_epoch{epoch+1}.pt", best_loss=run_best_loss)
        if (epoch+1) % PUSH_WEIGHTS_EVERY == 0:
            save_model(model, optimizer, epoch+1, cfg, "myai.pt", best_loss=run_best_loss)
            force_push_now("tokenizer.json")
        delete_mid_epoch()

        if phase_done: break

    return run_best_loss, last_epoch, stop_training


# ═══════════════════════════════════════════════════
#   GENERALIZATION PASS
#   Runs after a phase hits ≤ 0.50 loss.
#   Trains briefly on rephrased Q&A + unseen data so
#   the model learns to understand rather than recite.
#
#   What it does:
#     1. Takes Q&A pairs from the trained window
#     2. Rephrases questions 5 different ways (same answer)
#     3. Adds samples from OUTSIDE the current window (unseen)
#     4. Shuffles everything and trains for ~200 steps at low LR
#   Result: model stops pattern-matching word-for-word and starts
#   generalizing to different phrasings of the same concept.
# ═══════════════════════════════════════════════════

import random as _random

def _rephrase_pair(text):
    """Rewrite a Q&A pair using a different template."""
    lines = text.strip().split("\n")
    q, a = "", ""
    for ln in lines:
        if ln.lower().startswith("question:"): q = ln[9:].strip()
        elif ln.lower().startswith("answer:"):  a = ln[7:].strip()
    if not q or not a:
        return text
    templates = [
        f"Q: {q}\nA: {a}",
        f"Tell me: {q}\n{a}",
        f"{q}\nAnswer: {a}",
        f"Question — {q}\nReply — {a}",
        f"User: {q}\nAssistant: {a}",
    ]
    return _random.choice(templates)

def run_generalization_pass(model, optimizer, loss_fn, tok, big_text,
                            current_chars, device, cfg, n_steps=300):
    """
    Short anti-memorization pass after a phase completes.
    Returns avg loss of the pass (informational only).
    """
    print(f"\n  [generalization] Anti-memorization pass ({n_steps} steps)...")
    model.train()

    window   = big_text[:current_chars]
    all_text = big_text

    # Extract Q&A pairs from trained window
    pairs = [p.strip() for p in window.split("\n\n")
             if "Question:" in p and "Answer:" in p]

    if not pairs:
        print("  [generalization] No Q&A pairs found — skipping")
        return float("inf")

    # Build pool: rephrased known + unseen data
    rephrased = [_rephrase_pair(p) for p in pairs]

    pool = list(rephrased)
    if current_chars < len(all_text):
        outside_text = all_text[current_chars:]
        outside_pairs = [p.strip() for p in outside_text.split("\n\n")
                         if "Question:" in p and "Answer:" in p]
        # 70% rephrased + 30% unseen
        n_out = max(10, len(outside_pairs) // 3)
        pool += outside_pairs[:n_out]

    _random.shuffle(pool)

    seq_len = cfg.seq_len
    # Use a lower LR for the generalization pass to avoid catastrophic forgetting
    gen_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    total_loss = 0.0; done = 0; step = 0
    while step < n_steps and pool:
        # Grab a mini-batch of 4 pairs
        batch_texts = [pool[(step + k) % len(pool)] for k in range(4)]
        step += 4

        ids_batch = []
        for t in batch_texts:
            ids = tok.encode(t)
            for i in range(0, len(ids) - seq_len, seq_len // 2):
                chunk = ids[i:i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    ids_batch.append(chunk)
        if not ids_batch:
            continue

        _random.shuffle(ids_batch)
        batch_t = torch.tensor(ids_batch[:8], dtype=torch.long).to(device)
        x, y   = batch_t[:, :-1], batch_t[:, 1:]

        gen_optimizer.zero_grad()
        logits = model(x)
        loss   = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        gen_optimizer.step()

        total_loss += loss.item(); done += 1

    avg = total_loss / max(done, 1)
    print(f"  [generalization] Done — avg loss: {avg:.4f}  ({done} mini-steps, "
          f"{len(pool)} pairs in pool)")
    return avg


def train_cpu(tok, big_text, data_hash, resume_checkpoint, is_mid_epoch):
    global stop_training
    device       = torch.device("cpu")
    full_dataset = TextDataset(big_text, tok)
    total_chars  = len(big_text)
    cpu_batch    = min(BATCH_SIZE, 8)

    # ── Training budget ────────────────────────────
    current_chars = get_or_ask_budget(total_chars)
    budget_max    = get_budget_max(total_chars)

    # ── Phase training setup (always enabled) ─────
    phase_state  = get_phase_config(budget_max)
    ph_size      = phase_state["phase_size"]
    ph_total     = phase_state["total_phases"]
    ph_current   = phase_state["current_phase"]
    ph_done      = phase_state["chars_done"]

    # ── Model setup ────────────────────────────────
    cfg = Config(); cfg.vocab_size = tok.vocab_size
    for k, v in SIZE_CONFIGS[MODEL_SIZE].items(): setattr(cfg, k, v)

    model     = MyAI(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)

    actual_start = START_EPOCH; start_step = 0; run_best_loss = float("inf")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        actual_start, start_step = apply_ckpt(ckpt, model, optimizer, cfg, device, interactive=True)
        run_best_loss = ckpt.get("best_loss", float("inf"))
        label    = "mid-epoch" if is_mid_epoch else resume_checkpoint
        best_str = f"{run_best_loss:.4f}" if run_best_loss < float("inf") else "n/a"
        print(f"Resumed: {label} (epoch={actual_start+1}, step={start_step}, best_loss={best_str})")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, last_epoch=actual_start - 1 if actual_start > 0 else -1)

    print(f"Model: {MODEL_SIZE} | {model.count_params():,} params | batch:{cpu_batch} | lr:{scheduler.get_last_lr()[0]:.6f}")
    if SERVER_URL and GRAD_PULL_EVERY > 0:
        start_wg_thread(); print("Worker blending: ON")
    print()

    t0 = time.time(); epoch_times = []; last_epoch = actual_start

    # ══════════════════════════════════════════════
    #   CUMULATIVE PHASE TRAINING LOOP
    #   Phase N trains on N×100k chars (all previous + new chunk).
    #   Must reach ≤ 0.50 loss before phase N+1 starts.
    # ══════════════════════════════════════════════
    chars_done  = ph_done
    phase_tgt   = phase_state.get("target_loss", TARGET_LOSS_PER_PHASE)

    for phase_idx in range(ph_current, ph_total):
        if stop_training: break

        # Cumulative window: phase 1=100k, phase 2=200k, phase 3=300k ...
        chars_this_phase = min((phase_idx + 1) * ph_size, budget_max)
        if chars_this_phase <= 0: break

        new_added = chars_this_phase - chars_done
        label = f"[Phase {phase_idx+1}/{ph_total}] "

        print(f"\n{'='*62}")
        print(f"  {label}Window: {format_size(chars_this_phase)} chars total")
        if phase_idx > 0:
            print(f"  (keeping {format_size(chars_done)} already learned"
                  f" + {format_size(new_added)} new chars)")
        print(f"  Goal: loss ≤ {phase_tgt}  (keeps training until reached)")
        print(f"{'='*62}")

        loader, use_n = _make_loader_cpu(full_dataset, chars_this_phase, total_chars, cpu_batch)
        print(f"  Sequences in window: {use_n:,}")

        # Fresh cosine LR schedule each phase
        scheduler_phase = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCHS, last_epoch=-1)

        run_best_loss, last_epoch, stopped = _run_one_phase_cpu(
            model, optimizer, scheduler_phase, loss_fn, loader, cfg,
            device, data_hash, run_best_loss,
            0, 0, t0, epoch_times,
            phase_label=label, target_loss=phase_tgt)

        # ── Generalization pass: prevent recitation, promote understanding ──
        if run_best_loss <= phase_tgt and not stopped:
            run_generalization_pass(
                model, optimizer, loss_fn, tok, big_text,
                chars_this_phase, device, cfg)

        chars_done = chars_this_phase

        # Save checkpoint for this phase
        phase_ckpt = f"myai_phase{phase_idx+1}.pt"
        save_model(model, optimizer, last_epoch+1, cfg, phase_ckpt,
                   best_loss=run_best_loss, trained_chars=chars_done)
        save_model(model, optimizer, last_epoch+1, cfg, "myai.pt",
                   best_loss=run_best_loss, trained_chars=chars_done)
        force_push_now("tokenizer.json")

        phase_state["current_phase"] = phase_idx + 1
        phase_state["chars_done"]    = chars_done
        save_phase_state(phase_state)

        pct    = chars_done / budget_max * 100
        status = f"✓ loss={run_best_loss:.4f} ≤ {phase_tgt}" if run_best_loss <= phase_tgt \
                 else f"loss={run_best_loss:.4f} (target {phase_tgt} not yet met)"
        print(f"\n  Phase {phase_idx+1}/{ph_total} done — "
              f"{format_size(chars_done)}/{format_size(budget_max)} ({pct:.0f}%) | {status}")

        if not stopped and phase_idx + 1 < ph_total:
            next_window = min((phase_idx + 2) * ph_size, budget_max)
            print(f"  Next: expand to {format_size(next_window)} chars "
                  f"(+{format_size(next_window - chars_done)} new)")

        if stopped: break

    # All phases complete
    if not stop_training:
        clear_phase_state()
        print(f"\n✓ All {ph_total} phases complete!")
        print(f"  Total trained: {format_size(chars_done)} chars")
        print(f"  Best loss    : {run_best_loss:.4f}")

    current_chars = chars_done

    # ── Final save ─────────────────────────────────
    stop_wg_thread(); stop_sync_watcher()
    print("\nSaving final model...")
    save_model(model, optimizer, last_epoch+1, cfg, "myai.pt",
               best_loss=run_best_loss, trained_chars=current_chars)
    delete_mid_epoch()
    if SERVER_URL:
        push_stats(last_epoch+1, EPOCHS, 0, 0, time.time()-t0, 0, "Training complete!")

    if not stop_training:
        save_training_complete(current_chars, total_chars, run_best_loss, data_hash)
        print(f"\n✓ Training complete!")
        print(f"  Trained : {format_size(current_chars)} / {format_size(total_chars)} chars")
        print(f"  Best loss: {run_best_loss:.4f}")
        print(f"  Run again to continue from next phase, or download more data first.")
        # Upload all state files so download_results.py can grab everything
        upload_state_files()

    print(f"Done! {fmt(time.time()-t0)}")

# ═══════════════════════════════════════════════════
#   GPU TRAINING
# ═══════════════════════════════════════════════════

def train_gpu(rank, world_size, vram_list, resume_checkpoint, data_hash, is_mid_epoch):
    global stop_training
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank); device = torch.device(f"cuda:{rank}"); is_main = rank == 0

    big_text = load_data()
    tok = Tokenizer(); tok.load("tokenizer.json")
    full_dataset = TextDataset(big_text, tok)
    total_chars  = len(big_text)

    # Budget (Kaggle auto-detects full; otherwise read from file)
    if is_main:
        current_chars = get_or_ask_budget(total_chars)
        budget_max    = get_budget_max(total_chars)
        # On Kaggle: auto-setup phase training if not already configured
        if is_kaggle():
            _ps = load_phase_state()
            if _ps is None or not _ps.get("enabled"):
                get_phase_config(budget_max)   # auto-creates Kaggle phase state
        bc = torch.tensor([current_chars, budget_max], dtype=torch.long, device=device)
    else:
        bc = torch.zeros(2, dtype=torch.long, device=device)
    dist.broadcast(bc, src=0)
    current_chars = bc[0].item()
    budget_max    = bc[1].item()

    full_n = len(full_dataset)
    use_n  = max(1, int(full_n * current_chars / total_chars)) if current_chars < total_chars else full_n
    dataset = Subset(full_dataset, range(use_n)) if use_n < full_n else full_dataset

    cfg = Config(); cfg.vocab_size = tok.vocab_size
    for k, v in SIZE_CONFIGS[MODEL_SIZE].items(): setattr(cfg, k, v)

    raw_model = MyAI(cfg).to(device)
    safe      = probe_safe_batch(raw_model, device, vram_list[rank], cfg.seq_len)
    my_batch  = max(1, min(BATCH_SIZE // world_size, safe))

    bt = torch.tensor([my_batch], dtype=torch.long, device=device)
    all_bt = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_bt, bt)
    all_sizes = [b.item() for b in all_bt]

    if is_main:
        print(f"\nBatch sizes (reserve={VRAM_RESERVE_GB}GB/GPU):")
        for i, (b, v) in enumerate(zip(all_sizes, vram_list)):
            print(f"  GPU {i}: {v:.1f}GB → batch={b}")
        print(f"Dataset: {use_n:,} seqs\n")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader  = DataLoader(dataset, batch_size=my_batch, sampler=sampler, num_workers=0, pin_memory=True)
    total_steps = len(loader)

    model     = DDP(raw_model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
    scaler    = torch.amp.GradScaler("cuda")

    actual_start = START_EPOCH; start_step = 0; run_best_loss = float("inf")

    if is_main and resume_checkpoint and os.path.exists(resume_checkpoint):
        ckpt = torch.load(resume_checkpoint, map_location=device)
        actual_start, start_step = apply_ckpt(ckpt, model, optimizer, cfg, device)
        run_best_loss = ckpt.get("best_loss", float("inf"))
        label = "mid-epoch" if is_mid_epoch else resume_checkpoint
        best_str = f"{run_best_loss:.4f}" if run_best_loss < float("inf") else "n/a"
        print(f"Resumed: {label} (epoch={actual_start+1}, step={start_step}, best_loss={best_str})")

    info = torch.tensor([actual_start, start_step], dtype=torch.long, device=device)
    dist.broadcast(info, src=0); actual_start, start_step = info[0].item(), info[1].item()

    # ── Fix: use last_epoch in constructor, no dummy scheduler.step() calls ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, last_epoch=actual_start - 1 if actual_start > 0 else -1)

    if is_main:
        print(f"Model: {MODEL_SIZE} | {model.module.count_params():,} params | lr:{scheduler.get_last_lr()[0]:.6f}")
        if SERVER_URL and GRAD_PULL_EVERY > 0:
            start_wg_thread()
            print(f"Worker blending: ON (blend={WORKER_GRAD_BLEND})")
        print()

    # Load phase target loss (GPU reads it from the file saved by main process)
    _ps = load_phase_state()
    phase_target_loss = _ps.get("target_loss", TARGET_LOSS_PER_PHASE) if _ps and _ps.get("enabled") else 0.0
    if is_main and phase_target_loss > 0:
        print(f"  Phase target loss: ≤ {phase_target_loss} (stops early when reached)")

    t0 = time.time(); epoch_times = []; last_epoch = actual_start; w_applies = 0

    for epoch in range(actual_start, EPOCHS):
        if stop_training: break
        last_epoch = epoch; sampler.set_epoch(epoch)
        ep_start = time.time(); total_loss = 0.0; ep_steps = 0
        skip_to = start_step if epoch == actual_start else 0; start_step = 0

        for i, batch in enumerate(loader):
            if i < skip_to: continue
            if stop_training:
                if is_main:
                    save_mid_epoch(model.module, optimizer, epoch, i, cfg, data_hash, run_best_loss)
                    print(f"Mid-epoch saved (epoch={epoch+1}, step={i})")
                break

            batch = batch.to(device); x, y = batch[:, :-1], batch[:, 1:]
            with torch.amp.autocast("cuda"):
                logits = model(x)
                loss   = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if is_main and SERVER_URL and GRAD_PULL_EVERY > 0 and (i+1) % GRAD_PULL_EVERY == 0:
                if blend_worker_grads(model): w_applies += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item(); ep_steps += 1

            if is_main and MID_EPOCH_EVERY > 0 and (i+1) % MID_EPOCH_EVERY == 0:
                save_mid_epoch(model.module, optimizer, epoch, i+1, cfg, data_hash, run_best_loss)

            # ── Every 10 steps: push stats + model to server (silent, background) ──
            if is_main and SERVER_URL and (i+1) % 10 == 0:
                ela10 = time.time()-ep_start
                eta10 = ela10/max(i+1-skip_to,1)*(total_steps-i-1)
                msg10 = f"Epoch {epoch+1} | {i+1}/{total_steps} ({(i+1)/total_steps*100:.0f}%) | loss:{loss.item():.4f}"
                threading.Thread(
                    target=push_stats,
                    args=(epoch+1, EPOCHS, loss.item(), scheduler.get_last_lr()[0],
                          time.time()-t0, eta10, msg10),
                    kwargs={"step": i+1, "total_steps": total_steps},
                    daemon=True
                ).start()
                threading.Thread(
                    target=lambda m=model.module, o=optimizer, e=epoch, s=i+1, c=cfg, bl=run_best_loss:
                        _push_model_microstep(m, o, e, s, c, bl),
                    daemon=True
                ).start()

            # ── Every 50 steps: print to console ──
            if is_main and (i+1) % 50 == 0:
                pct = (i+1)/total_steps*100; ela = time.time()-ep_start
                eta = ela/max(i+1-skip_to,1)*(total_steps-i-1)
                wi  = f" workers:{w_applies}" if w_applies else ""
                msg = f"Epoch {epoch+1} | {i+1}/{total_steps} ({pct:.0f}%) | loss:{loss.item():.4f}{wi}"
                print(msg+f" | eta:{fmt(eta)}")
                push_stats(epoch+1,EPOCHS,loss.item(),scheduler.get_last_lr()[0],
                           time.time()-t0,eta,msg,step=i+1,total_steps=total_steps)

        if stop_training: break

        ep_time = time.time()-ep_start; avg = total_loss/max(ep_steps,1)
        eta_tot = (sum(epoch_times+[ep_time])/(len(epoch_times)+1))*(EPOCHS-epoch-1)
        epoch_times.append(ep_time)
        scheduler.step()

        if is_main:
            if w_applies: print(f"  [workers] blended {w_applies} times this epoch")
            w_applies = 0
            if SERVER_URL: flush_worker_grads(model.module, optimizer)
            if avg < run_best_loss: run_best_loss = avg

            # ── Phase target loss check ────────────────────────────────────
            if phase_target_loss > 0 and avg <= phase_target_loss:
                print(f"  ✓ Target loss reached ({avg:.4f} ≤ {phase_target_loss}) — phase complete!")
                stop_training = True   # signal all ranks to stop this phase

            msg = (f"Epoch {epoch+1}/{EPOCHS} loss:{avg:.4f} "
                   f"lr:{scheduler.get_last_lr()[0]:.6f} "
                   f"took:{fmt(ep_time)} eta:{fmt(eta_tot)}")
            print(msg)
            push_stats(epoch+1,EPOCHS,avg,scheduler.get_last_lr()[0],
                       time.time()-t0,eta_tot,msg,step=total_steps,total_steps=total_steps)

            if (epoch+1) % SAVE_EVERY == 0:
                save_model(model,optimizer,epoch+1,cfg,f"myai_epoch{epoch+1}.pt",best_loss=run_best_loss)

            # Save myai.pt — file watcher auto-pushes it to server
            if (epoch+1) % PUSH_WEIGHTS_EVERY == 0:
                save_model(model,optimizer,epoch+1,cfg,"myai.pt",best_loss=run_best_loss)
                force_push_now("tokenizer.json")

            delete_mid_epoch()

            # ── Progressive expansion ──────────────
            if PROGRESSIVE and avg < TARGET_LOSS and current_chars < budget_max:
                new_chars = min(int(current_chars * PROGRESSIVE_FACTOR), budget_max)
                full_n2   = len(full_dataset)
                use_n2    = max(1, int(full_n2 * new_chars / total_chars)) if new_chars < total_chars else full_n2
                dataset   = Subset(full_dataset, range(use_n2)) if use_n2 < full_n2 else full_dataset
                sampler   = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
                loader    = DataLoader(dataset, batch_size=my_batch, sampler=sampler,
                                       num_workers=0, pin_memory=True)
                total_steps = len(loader)
                print(f"  ↑ Progressive: loss {avg:.4f} < {TARGET_LOSS} — "
                      f"expanding {format_size(current_chars)} → {format_size(new_chars)} "
                      f"({use_n2:,} seqs)")
                current_chars = new_chars
                update_budget_current(current_chars)

    if is_main:
        stop_wg_thread(); stop_sync_watcher()
        print("\nSaving final model...")
        save_model(model,optimizer,last_epoch+1,cfg,"myai.pt",best_loss=run_best_loss)
        delete_mid_epoch()
        if SERVER_URL:
            push_stats(last_epoch+1,EPOCHS,0,0,time.time()-t0,0,"Training complete!")
        if not stop_training:
            save_training_complete(current_chars, total_chars, run_best_loss, data_hash)
            asc = load_autoscale()
            print(f"\n✓ Training complete! Trained {format_size(current_chars)} / {format_size(total_chars)} chars")
            print(f"  Best loss: {run_best_loss:.4f}")
            if asc and asc.get("enabled"):
                step = asc["step_chars"]
                nxt  = min(current_chars + step, total_chars)
                print(f"  Next run will auto-expand to {format_size(nxt)} chars (+{format_size(step)})")
            else:
                print(f"  Run again to train more chars, or run with --extend to add downloaded data.")
            upload_state_files()
        print(f"Done! {fmt(time.time()-t0)}")

    dist.destroy_process_group()

# ═══════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.backends.cudnn.benchmark   = True

    print("MyAI Trainer"); print("=" * 40)
    if SERVER_URL: print(f"Server: {SERVER_URL}")

    device_type, num_gpus = detect_device()

    print("\nChecking training data...")
    data_changed = check_data_changed()
    data_hash    = compute_data_hash(find_data_path())
    big_text     = load_data()
    print(f"Text: {len(big_text):,} chars")

    if data_changed or not os.path.exists("tokenizer.json"):
        print("Building tokenizer...")
        tok = Tokenizer(); tok.build_vocab([big_text], max_vocab=VOCAB_SIZE)
        tok.save("tokenizer.json"); print(f"Tokenizer: {tok.vocab_size} tokens")
    else:
        tok = Tokenizer(); tok.load("tokenizer.json")
        print(f"Tokenizer: {tok.vocab_size} tokens (reused)")

    resume, is_mid = pick_resume_checkpoint(data_changed, data_hash)

    # ── Start background file watcher ──────────────
    # Instantly pushes tokenizer.json and myai.pt to server
    # whenever they change on disk — no manual sync needed.
    start_sync_watcher()

    # Force-push tokenizer immediately so workers can connect right away.
    # This runs regardless of data_changed so the server always has it.
    if os.path.exists("tokenizer.json"):
        force_push_now("tokenizer.json")

    # ── Upload training data to server so web/script workers can get batches ──
    # This only uploads if the server doesn't have it yet, or data changed.
    if SERVER_URL:
        data_path = find_data_path()
        if data_path:
            print("Checking if server has training data...")
            try:
                r = server_get("/has_training_data", timeout=8)
                server_has_data = r and r.status_code == 200 and r.json().get("ok")
            except Exception:
                server_has_data = False

            if not server_has_data or data_changed:
                fsize = os.path.getsize(data_path)
                print(f"Uploading training data to server ({fsize/1024/1024:.1f} MB) — workers need this...")
                try:
                    with open(data_path, "rb") as f:
                        r = requests.post(f"{SERVER_URL}/training_data", data=f.read(),
                                          headers={"X-Secret-Key": SECRET_KEY}, timeout=300)
                    if r and r.status_code == 200:
                        print("Training data uploaded — workers can now get batches!")
                    else:
                        print(f"Training data upload failed (HTTP {r.status_code if r else '?'}) — workers may have no data")
                except Exception as e:
                    print(f"Training data upload error: {e} — workers may have no data")
            else:
                print("Server already has training data — skipping upload")

    print()
    if device_type == "cpu":
        train_cpu(tok, big_text, data_hash, resume, is_mid)
    elif num_gpus == 1:
        print("1 GPU — starting...")
        mp.spawn(train_gpu, args=(1, get_gpu_info(), resume, data_hash, is_mid), nprocs=1, join=True)
    else:
        print(f"{num_gpus} GPUs — DDP...")
        mp.spawn(train_gpu, args=(num_gpus, get_gpu_info(), resume, data_hash, is_mid),
                 nprocs=num_gpus, join=True)
