#!/usr/bin/env python3
# download_results.py
# ─────────────────────────────────────────────────────────────────────────────
#   Downloads ALL training result files from the server after training ends.
#   Run this on your local PC once train.py has finished on Kaggle / elsewhere.
#
#   Files downloaded (if they exist on server):
#     myai.pt                  — main model weights
#     myai_epochX.pt           — last epoch checkpoint (e.g. myai_epoch30.pt)
#     tokenizer.json           — vocabulary
#     .best_loss               — best loss tracker (hidden file)
#     best_loss                — same, without dot prefix
#     .data_hash               — data fingerprint (hidden file)
#     data_hash                — same, without dot prefix
#     .training_budget.json    — how much data was used
#     training_budget.json     — same, without dot prefix
#     .phase_state.json        — which phase we're on
#     phase_state.json         — same, without dot prefix
#     .training_complete.json  — training-done marker
#     training_complete.json   — same, without dot prefix
#
#   Rule: if BOTH the dot and non-dot version exist → download both.
#         If only one exists → download whichever one exists.
# ─────────────────────────────────────────────────────────────────────────────

import sys, subprocess, os, json, time

# ── Auto-install requests ─────────────────────────────────────────────────────
try:
    import requests
except ImportError:
    print("Installing requests...")
    for flags in [["--break-system-packages", "-q"], ["-q"]]:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "requests"] + flags,
                           capture_output=True, text=True)
        if r.returncode == 0: break
    import requests

# ─────────────────────────────────────────────────────────────────────────────
#   CONFIGURATION  (must match train.py)
# ─────────────────────────────────────────────────────────────────────────────

SERVER_URL = "https://eh-production.up.railway.app"
SECRET_KEY = "Dsadasdsefgtgtlubiemlodydsadasdseflubiemlody1bekekejroliwer2011elo%5dfdsfdsk"

# ─────────────────────────────────────────────────────────────────────────────
#   FILE LIST
#   Each entry is (server_route, local_filename, is_binary, secret_needed)
#   For dot/no-dot pairs they're listed separately — the server checks each.
# ─────────────────────────────────────────────────────────────────────────────

# Files served via dedicated routes
DEDICATED_ROUTES = [
    ("/model",     "myai.pt",         True,  False),
    ("/tokenizer", "tokenizer.json",  False, False),
]

# Generic state files served via /file/<name> route
# Listed as (server_name, local_name, is_binary)
# Dot and no-dot versions are both attempted — download whichever exist.
STATE_FILES = [
    # dot version          no-dot version
    (".best_loss",              "best_loss"),
    (".data_hash",              "data_hash"),
    (".training_budget.json",   "training_budget.json"),
    (".phase_state.json",       "phase_state.json"),
    (".training_complete.json", "training_complete.json"),
    (".autoscale.json",         "autoscale.json"),
]

# ─────────────────────────────────────────────────────────────────────────────

def fmt_size(n):
    if n >= 1024*1024: return f"{n/1024/1024:.1f} MB"
    if n >= 1024:      return f"{n/1024:.0f} KB"
    return f"{n} B"

def download_file(url, local_path, headers=None, binary=True, label=None):
    """Download url → local_path. Returns True on success."""
    label = label or os.path.basename(local_path)
    try:
        r = requests.get(url, headers=headers or {}, timeout=120, stream=binary)
        if r.status_code == 404:
            return False   # doesn't exist — not an error
        if r.status_code != 200:
            print(f"  ✗  {label}  (HTTP {r.status_code})")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

        if binary:
            total = int(r.headers.get("Content-Length", 0))
            got   = 0
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=512*1024):
                    f.write(chunk); got += len(chunk)
                    if total > 1024*1024:  # progress bar only for large files
                        pct = got/total*100 if total else 0
                        print(f"\r  ↓  {label}  {fmt_size(got)}/{fmt_size(total)} ({pct:.0f}%)",
                              end="", flush=True)
            if total > 1024*1024: print()
        else:
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(r.text)

        size = os.path.getsize(local_path)
        print(f"  ✓  {label}  ({fmt_size(size)})")
        return True

    except Exception as e:
        print(f"  ✗  {label}  ERROR: {e}")
        return False

def check_exists(url, headers=None):
    """HEAD request to check if a file exists on the server."""
    try:
        r = requests.head(url, headers=headers or {}, timeout=10)
        return r.status_code == 200
    except Exception:
        # HEAD not supported — try GET with tiny timeout
        try:
            r = requests.get(url, headers=headers or {}, timeout=10, stream=True)
            r.close()
            return r.status_code == 200
        except Exception:
            return False

def main():
    print("=" * 56)
    print("  MyAI — Download Training Results")
    print("=" * 56)
    print(f"  Server: {SERVER_URL}\n")

    hdrs_auth = {"X-Secret-Key": SECRET_KEY}
    downloaded = []; skipped = []; failed = []

    # ── 1. Main model ─────────────────────────────────────────────────────────
    print("── Model weights ───────────────────────────────────────")
    ok = download_file(f"{SERVER_URL}/model", "myai.pt", binary=True)
    if ok:   downloaded.append("myai.pt")
    else:    print(f"  –  myai.pt  (not on server yet)"); skipped.append("myai.pt")

    # ── 2. Latest epoch checkpoint (myai_epochX.pt) ───────────────────────────
    print("\n── Epoch checkpoint ────────────────────────────────────")
    try:
        r = requests.get(f"{SERVER_URL}/latest_epoch_ckpt",
                         headers=hdrs_auth, timeout=15)
        if r.status_code == 200:
            data = r.json()
            fname = data.get("filename")
            if fname:
                ok = download_file(f"{SERVER_URL}/file/{fname}", fname, binary=True)
                if ok: downloaded.append(fname)
                else:  failed.append(fname)
            else:
                print("  –  No epoch checkpoint on server")
                skipped.append("myai_epochX.pt")
        else:
            print("  –  No epoch checkpoint on server")
            skipped.append("myai_epochX.pt")
    except Exception as e:
        print(f"  ✗  Epoch checkpoint error: {e}")
        failed.append("myai_epochX.pt")

    # ── 3. Tokenizer ──────────────────────────────────────────────────────────
    print("\n── Tokenizer ───────────────────────────────────────────")
    try:
        r = requests.get(f"{SERVER_URL}/tokenizer", timeout=30)
        if r.status_code == 200:
            tok = r.json()
            if "word2id" in tok:
                with open("tokenizer.json", "w") as f: json.dump(tok, f)
                size = os.path.getsize("tokenizer.json")
                print(f"  ✓  tokenizer.json  ({fmt_size(size)})")
                downloaded.append("tokenizer.json")
            else:
                print(f"  –  tokenizer.json  (not ready: {tok.get('error','?')})")
                skipped.append("tokenizer.json")
        else:
            print(f"  –  tokenizer.json  (HTTP {r.status_code})")
            skipped.append("tokenizer.json")
    except Exception as e:
        print(f"  ✗  tokenizer.json  ERROR: {e}")
        failed.append("tokenizer.json")

    # ── 4. State files (dot + no-dot versions) ────────────────────────────────
    print("\n── State & config files ────────────────────────────────")
    for dot_name, plain_name in STATE_FILES:
        # Check both versions
        dot_url   = f"{SERVER_URL}/file/{dot_name}"
        plain_url = f"{SERVER_URL}/file/{plain_name}"

        is_binary = dot_name.endswith(".pt")
        is_json   = dot_name.endswith(".json")

        dot_exists   = check_exists(dot_url,   headers=hdrs_auth)
        plain_exists = check_exists(plain_url, headers=hdrs_auth)

        if not dot_exists and not plain_exists:
            print(f"  –  {plain_name}  (not on server)")
            skipped.append(plain_name)
            continue

        # Download whichever versions exist
        if dot_exists:
            ok = download_file(dot_url, dot_name,
                               headers=hdrs_auth, binary=is_binary,
                               label=dot_name)
            if ok: downloaded.append(dot_name)
            else:  failed.append(dot_name)

        if plain_exists:
            ok = download_file(plain_url, plain_name,
                               headers=hdrs_auth, binary=is_binary,
                               label=plain_name)
            if ok: downloaded.append(plain_name)
            else:  failed.append(plain_name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 56)
    print(f"  ✓ Downloaded : {len(downloaded)} file(s)")
    if skipped: print(f"  –  Not found : {len(skipped)} (not on server yet)")
    if failed:  print(f"  ✗ Failed     : {len(failed)}  — {failed}")

    if downloaded:
        print(f"\n  Files saved to: {os.getcwd()}")
        if "myai.pt" in downloaded:
            try:
                import torch
                ckpt = torch.load("myai.pt", map_location="cpu")
                epoch = ckpt.get("epoch", "?")
                loss  = ckpt.get("best_loss", None)
                chars = ckpt.get("trained_chars", None)
                print(f"\n  Model summary:")
                print(f"    Epoch      : {epoch}")
                print(f"    Best loss  : {f'{loss:.4f}' if loss else 'n/a'}")
                if chars:
                    mb = chars / 1_000_000
                    print(f"    Trained on : {mb:.2f}m chars")
            except Exception:
                pass
        print(f"\n  Run: python chat.py")
    else:
        print("\n  Nothing downloaded. Is train.py still running?")
        print("  Re-run this script when training is complete.")

    print("=" * 56)

if __name__ == "__main__":
    main()
