# kaggle_setup.py
# ═══════════════════════════════════════════════════════════════════════════════
#   MyAI — Kaggle Setup + Auto GitHub Save
#   Copies all required files from your Kaggle dataset to /kaggle/working,
#   trains, then pushes ALL important files to GitHub when done
#   (whether training finished, crashed, or Kaggle ran out of GPU time).
#
#   FIRST TIME SETUP:
#     1. Go to GitHub → Settings → Developer Settings → Personal Access Tokens
#        → Tokens (classic) → Generate new token
#        → Give it "repo" scope → copy the token
#     2. In Kaggle notebook → Add-ons → Secrets → Add secret:
#           Name:  GITHUB_TOKEN
#           Value: (paste your token here)
#     3. Upload all your files to your Kaggle dataset (see REQUIRED_FILES below)
#     4. Run this script — everything else is automatic
# ═══════════════════════════════════════════════════════════════════════════════

import shutil, os, sys, json, glob, subprocess, signal, threading, time, atexit

# ── Configuration ──────────────────────────────────────────────────────────────
BASE    = '/kaggle/input/datasets/oliwierpruchnik/my-ai-model2'
WORKING = '/kaggle/working'

GITHUB_REPO     = 'https://github.com/fagatabez/keggle_save'
GITHUB_USERNAME = 'fagatabez'
GITHUB_BRANCH   = 'main'

# Files pushed to GitHub after training (in priority order)
FILES_TO_SAVE = [
    # ── Weights (most important) ──────────────────
    'myai.pt',
    'myai_midepoch.pt',
    # ── Tokenizer ─────────────────────────────────
    'tokenizer.json',
    # ── Code ──────────────────────────────────────
    'model.py',
    'tokenizer.py',
    'data.py',
    'train.py',
    'chat.py',
    'download_data.py',
    'kaggle_setup.py',
    # ── State / settings ──────────────────────────
    '.training_budget.json',
    '.autoscale.json',
    '.best_loss',
    '.data_hash',
    '.training_complete.json',
]

# Also push any myai_epoch*.pt checkpoints found in /kaggle/working
PUSH_EPOCH_CKPTS = True

# Push training data to GitHub?
# Usually too large (GitHub limit = 100 MB per file).
# Set True only if your training_data.txt.gz is under ~90 MB.
PUSH_TRAINING_DATA  = False
TRAINING_DATA_FILES = ['training_data.txt.gz', 'training_data.txt']

# Auto-save to GitHub every N minutes while training runs.
PERIODIC_SAVE_MINUTES = 5

# ── Files to copy from Kaggle dataset ─────────────────────────────────────────
REQUIRED_FILES = ['model.py', 'tokenizer.py', 'data.py', 'train.py']

TRAINING_DATA_OPTIONS = ['training_data.txt.gz', 'training_data.txt']

OPTIONAL_FILES = [
    'myai.pt',
    'tokenizer.json',
    'myai_midepoch.pt',
    '.training_budget.json',
    '.autoscale.json',
    '.best_loss',
    '.data_hash',
    '.training_complete.json',
    'chat.py',
    'download_data.py',
]

COPY_LATEST_EPOCH_CKPT = True

# ═══════════════════════════════════════════════════════════════════════════════
#   GITHUB PUSH
# ═══════════════════════════════════════════════════════════════════════════════

_github_push_lock = threading.Lock()
_push_count       = 0
_push_disabled    = False   # set True after a permanent failure — stops all retries

def get_github_token():
    """Read GitHub token from Kaggle Secrets or environment variable."""
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        if token and token.strip():
            return token.strip()
    except Exception:
        pass
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    return token if token else None

def run_cmd(cmd, cwd=None):
    """Run shell command, return (success: bool, output: str)."""
    result = subprocess.run(cmd, shell=True, cwd=cwd,
                            capture_output=True, text=True)
    return result.returncode == 0, (result.stdout + result.stderr).strip()

def push_to_github(reason="training finished"):
    """
    Clone repo → copy working files → commit → push.
    Thread-safe. Called from atexit, signal handlers, periodic thread, and finally block.
    Permanently disabled after a 403 / auth error so it never spams retries.
    """
    global _push_count, _push_disabled

    # ── Skip immediately if already known to be broken ─────────────────────────
    if _push_disabled:
        return False

    with _github_push_lock:
        _push_count += 1
        current_push = _push_count

    print("\n" + "═" * 60)
    print(f"  GitHub Save #{current_push}  —  {reason}")
    print("═" * 60)

    # ── Token ──────────────────────────────────────────────────────────────────
    token = get_github_token()
    if not token:
        print("✗ GITHUB_TOKEN not found — cannot save!")
        print("  Add it: Kaggle → Add-ons → Secrets → GITHUB_TOKEN")
        _push_disabled = True
        print("  GitHub auto-save DISABLED for this run (no token).")
        return False

    repo_url = GITHUB_REPO.replace(
        "https://github.com/",
        f"https://{GITHUB_USERNAME}:{token}@github.com/"
    )

    # ── Clone ──────────────────────────────────────────────────────────────────
    clone_dir = '/kaggle/working/_github_save'
    if os.path.exists(clone_dir):
        shutil.rmtree(clone_dir, ignore_errors=True)

    print(f"Cloning {GITHUB_REPO} ...")
    ok, out = run_cmd(f'git clone --depth=1 --branch {GITHUB_BRANCH} {repo_url} {clone_dir}')
    if not ok:
        print(f"  Branch '{GITHUB_BRANCH}' not found — creating it...")
        ok, out = run_cmd(f'git clone --depth=1 {repo_url} {clone_dir}')
        if not ok:
            # Check for auth failure — permanent, no point retrying
            if "403" in out or "Permission" in out or "authentication" in out.lower():
                _push_disabled = True
                print(f"\n✗ Auth error — GitHub pushes DISABLED for this run.")
                print("  Fix: regenerate your token with 'repo' scope, update Kaggle Secret.")
                return False
            print("  Repo appears empty — initialising fresh repo...")
            os.makedirs(clone_dir, exist_ok=True)
            run_cmd('git init', cwd=clone_dir)
            run_cmd(f'git remote add origin {repo_url}', cwd=clone_dir)
        run_cmd(f'git checkout -b {GITHUB_BRANCH}', cwd=clone_dir)

    run_cmd('git config user.email "kaggle@myai.bot"', cwd=clone_dir)
    run_cmd('git config user.name "MyAI Kaggle Bot"',  cwd=clone_dir)

    # ── Build file list ────────────────────────────────────────────────────────
    to_push = list(FILES_TO_SAVE)

    if PUSH_EPOCH_CKPTS:
        ckpts = sorted(
            glob.glob(os.path.join(WORKING, 'myai_epoch*.pt')),
            key=lambda p: int(
                os.path.basename(p).replace('myai_epoch','').replace('.pt','')
            ) if os.path.basename(p).replace('myai_epoch','').replace('.pt','').isdigit() else 0
        )
        to_push += [os.path.basename(p) for p in ckpts]

    if PUSH_TRAINING_DATA:
        to_push += TRAINING_DATA_FILES

    # ── Copy files ─────────────────────────────────────────────────────────────
    print("\nCopying files into repo:")
    copied = []; skipped = []
    for fname in to_push:
        src = os.path.join(WORKING, fname)
        if not os.path.exists(src):
            skipped.append(fname)
            continue
        size = os.path.getsize(src)
        if size > 95 * 1024 * 1024:
            print(f"  ⚠  {fname}  ({size/1024/1024:.0f} MB) — over GitHub 95MB limit, skipped")
            skipped.append(fname)
            continue
        try:
            shutil.copy2(src, os.path.join(clone_dir, fname))
            size_str = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.0f} KB"
            print(f"  ✓  {fname}  ({size_str})")
            copied.append(fname)
        except Exception as e:
            print(f"  ✗  {fname}  ERROR: {e}")
            skipped.append(fname)

    if not copied:
        print("No files to push.")
        return False

    if skipped:
        print(f"\n  Skipped: {', '.join(skipped)}")

    # ── Write save_info.json ───────────────────────────────────────────────────
    save_info = {
        "saved_at":      time.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "reason":        reason,
        "push_number":   current_push,
        "files_saved":   copied,
        "files_skipped": skipped,
    }
    if 'myai.pt' in copied:
        try:
            import torch
            ckpt = torch.load(os.path.join(WORKING, 'myai.pt'), map_location='cpu')
            save_info["epoch"]        = ckpt.get("epoch", "?")
            save_info["best_loss"]    = round(float(ckpt.get("best_loss", 0) or 0), 4)
            cfg = ckpt.get("config", {})
            save_info["model_config"] = {k: cfg.get(k) for k in
                                         ("embed_dim","num_layers","num_heads","vocab_size")}
        except Exception:
            pass

    with open(os.path.join(clone_dir, 'save_info.json'), 'w') as f:
        json.dump(save_info, f, indent=2)

    # ── Commit & push ──────────────────────────────────────────────────────────
    ts      = time.strftime("%Y-%m-%d %H:%M")
    loss    = save_info.get("best_loss", "?")
    epoch   = save_info.get("epoch", "?")
    message = f"[MyAI] {reason} | epoch={epoch} loss={loss} | {ts}"

    print(f"\nCommit: {message}")
    run_cmd('git add -A', cwd=clone_dir)
    ok, out = run_cmd(f'git commit -m "{message}"', cwd=clone_dir)
    if not ok and "nothing to commit" in out:
        print("  No changes since last save — skipping push.")
        return True

    print("Pushing...")
    ok, out = run_cmd(f'git push origin {GITHUB_BRANCH}', cwd=clone_dir)
    if ok:
        print(f"\n✓ Saved! → {GITHUB_REPO}/tree/{GITHUB_BRANCH}")
        print(f"  {len(copied)} files  |  push #{current_push}  |  {reason}")
        return True
    else:
        print(f"\n✗ Push failed:\n{out}")
        # ── Permanent auth failure → disable for rest of session ───────────────
        if "403" in out or "Permission" in out or "authentication" in out.lower() \
                or "denied" in out.lower():
            _push_disabled = True
            print("\n  ⛔ Auth error — GitHub auto-save DISABLED for this run.")
            print("  Fix your token, then re-run the notebook:")
            print("    1. GitHub → Settings → Developer settings → Tokens (classic)")
            print("    2. Generate new token with 'repo' scope checked")
            print("    3. Kaggle → Add-ons → Secrets → update GITHUB_TOKEN")
        else:
            print("  Will retry next save interval.")
        return False

# ── Periodic background saver ──────────────────────────────────────────────────
_training_running = True

def _periodic_saver():
    while _training_running:
        for _ in range(PERIODIC_SAVE_MINUTES * 60):
            if not _training_running:
                return
            time.sleep(1)
        if not _training_running:
            return
        if _push_disabled:
            return   # stop the thread entirely once disabled
        print(f"\n[auto-save] {PERIODIC_SAVE_MINUTES}min interval — pushing to GitHub...")
        push_to_github(reason=f"periodic save (every {PERIODIC_SAVE_MINUTES}min)")

_saver_thread = threading.Thread(target=_periodic_saver, daemon=True)

# ── Exit handlers — fire on ANY kind of stop ──────────────────────────────────
def _atexit_handler():
    if not _push_disabled:
        push_to_github(reason="session ended (atexit)")

def _signal_handler(sig, frame):
    global _training_running
    _training_running = False
    if not _push_disabled:
        signame = {signal.SIGTERM: "SIGTERM (timeout/kill)",
                   signal.SIGINT:  "SIGINT (Ctrl+C)"}.get(sig, f"signal {sig}")
        push_to_github(reason=signame)
    sys.exit(0)

atexit.register(_atexit_handler)
signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)

# ═══════════════════════════════════════════════════════════════════════════════
#   SETUP — Copy files from dataset → /kaggle/working
# ═══════════════════════════════════════════════════════════════════════════════

def copy_file(src, dst, label=""):
    try:
        shutil.copy(src, dst)
        size     = os.path.getsize(dst)
        size_str = f"{size/1024/1024:.1f} MB" if size > 1024*1024 else f"{size/1024:.0f} KB"
        print(f"  ✓  {label or os.path.basename(src)}  ({size_str})")
        return True
    except FileNotFoundError:
        print(f"  ✗  {label or os.path.basename(src)}  NOT FOUND in dataset")
        return False
    except Exception as e:
        print(f"  ✗  {label or os.path.basename(src)}  ERROR: {e}")
        return False

def find_in_dataset(filename):
    path = os.path.join(BASE, filename)
    return path if os.path.exists(path) else None

print("═" * 60)
print("  MyAI — Kaggle Setup + GitHub Auto-Save")
print("═" * 60)

if not os.path.exists(BASE):
    print(f"\n✗ Dataset not found: {BASE}")
    sys.exit(1)

print(f"\nDataset : {BASE}")
os.chdir(WORKING)
print(f"Working : {WORKING}\n")

# ── Required code ──────────────────────────────────────────────────────────────
print("── Required code files ─────────────────────────────────")
missing = []
for f in REQUIRED_FILES:
    src = find_in_dataset(f)
    if src: copy_file(src, os.path.join(WORKING, f))
    else:   missing.append(f); print(f"  ✗  {f}  MISSING")
if missing:
    print(f"\n✗ FATAL: missing required files: {missing}")
    sys.exit(1)

# ── Training data ──────────────────────────────────────────────────────────────
print("\n── Training data ───────────────────────────────────────")
data_copied = False
for data_file in TRAINING_DATA_OPTIONS:
    src = find_in_dataset(data_file)
    if src:
        copy_file(src, os.path.join(WORKING, data_file))
        data_copied = True; break
if not data_copied:
    print("  ✗ No training data found!"); sys.exit(1)

# ── Optional files ─────────────────────────────────────────────────────────────
print("\n── Checkpoint & state files ────────────────────────────")
found_checkpoint = False
for f in OPTIONAL_FILES:
    src = find_in_dataset(f)
    if src:
        copy_file(src, os.path.join(WORKING, f))
        if f == 'myai.pt': found_checkpoint = True
    else:
        print(f"  –  {f}  (not in dataset)")

if COPY_LATEST_EPOCH_CKPT:
    ckpts = sorted(
        glob.glob(os.path.join(BASE, 'myai_epoch*.pt')),
        key=lambda p: int(
            os.path.basename(p).replace('myai_epoch','').replace('.pt','')
        ) if os.path.basename(p).replace('myai_epoch','').replace('.pt','').isdigit() else 0
    )
    if ckpts:
        latest = ckpts[-1]
        copy_file(latest, os.path.join(WORKING, os.path.basename(latest)),
                  label=f"{os.path.basename(latest)} (latest epoch ckpt)")
    else:
        print("  –  myai_epoch*.pt  (none found)")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Summary ─────────────────────────────────────────────")
if found_checkpoint:
    try:
        import torch
        ckpt     = torch.load(os.path.join(WORKING, 'myai.pt'), map_location='cpu')
        epoch    = ckpt.get('epoch', '?')
        loss     = ckpt.get('best_loss', None)
        loss_str = f"{loss:.4f}" if loss is not None else "n/a"
        cfg      = ckpt.get('config', {})
        print(f"  Mode      : RESUME")
        print(f"  Epoch     : {epoch}")
        print(f"  Best loss : {loss_str}")
        if cfg:
            print(f"  Model     : embed={cfg.get('embed_dim')}  "
                  f"layers={cfg.get('num_layers')}  heads={cfg.get('num_heads')}")
    except Exception as e:
        print(f"  Mode      : RESUME  (could not read details: {e})")
else:
    print("  Mode      : FRESH START  (no myai.pt found in dataset)")
    print("  Tip: download myai.pt from GitHub after a run and re-upload to dataset")

token_ok = bool(get_github_token())
print(f"\n  GitHub    : {GITHUB_REPO}")
print(f"  Auto-save : every {PERIODIC_SAVE_MINUTES} min + on finish / crash / timeout")
print(f"  Token     : {'✓ GITHUB_TOKEN found' if token_ok else '✗ GITHUB_TOKEN MISSING — saves will fail'}")
if not token_ok:
    print("  → Add it:  Kaggle → Add-ons → Secrets → GITHUB_TOKEN")

print("\n" + "═" * 60)
print("  Starting training...")
print("═" * 60 + "\n")

# ── Start periodic background saver ───────────────────────────────────────────
_saver_thread.start()

# ═══════════════════════════════════════════════════════════════════════════════
#   TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

try:
    exit_code = os.system('python train.py')
    reason    = "training complete" if exit_code == 0 else f"training exited (code {exit_code})"
except Exception as e:
    reason = f"training crashed: {e}"
finally:
    _training_running = False
    push_to_github(reason=reason)
