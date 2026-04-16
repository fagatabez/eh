#!/usr/bin/env python3
# download_checkpoint.py
# Downloads your trained model from the server, then deletes the server copy.
# Run this on your local PC after train.py has synced to the server.

import sys, subprocess, os

# Auto-install requests if missing
try:
    import requests
except ImportError:
    print("Installing requests...")
    for flags in [["--break-system-packages","-q"],["-q"]]:
        r = subprocess.run([sys.executable,"-m","pip","install","requests"]+flags,
                           capture_output=True, text=True)
        if r.returncode == 0: break
    import requests

import json

SERVER_URL = "https://eh-production.up.railway.app"
SECRET_KEY = "Dsadasdsefgtgtlubiemlodydsadasdseflubiemlody1bekekejroliwer2011elo%5dfdsfdsk"
SAVE_AS    = "myai.pt"

def main():
    print(f"Checking {SERVER_URL} for model...")

    # ── Download model ─────────────────────────────
    try:
        r = requests.get(f"{SERVER_URL}/model", timeout=120, stream=True)
    except Exception as e:
        print(f"Connection failed: {e}"); sys.exit(1)

    if r.status_code == 404:
        print("No model on server yet. Run train.py first."); sys.exit(0)
    if r.status_code != 200:
        print(f"Server error HTTP {r.status_code}"); sys.exit(1)

    total = int(r.headers.get("Content-Length", 0))
    got   = 0
    with open(SAVE_AS, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk); got += len(chunk)
            if total:
                print(f"\r  {got/1024/1024:.1f}/{total/1024/1024:.1f} MB "
                      f"({got/total*100:.0f}%)", end="", flush=True)
    print(f"\nSaved: {SAVE_AS}  ({got/1024/1024:.1f} MB)")

    # ── Delete server copy ─────────────────────────
    print("Deleting from server (no duplicates)...")
    try:
        d = requests.delete(f"{SERVER_URL}/model",
            headers={"X-Secret-Key": SECRET_KEY}, timeout=15)
        if d.status_code == 200:   print("Server copy deleted.")
        elif d.status_code == 404: print("Already gone — OK.")
        else: print(f"Delete HTTP {d.status_code}")
    except Exception as e:
        print(f"Delete failed: {e}")

    # ── Download tokenizer ─────────────────────────
    print("\nDownloading tokenizer...")
    try:
        t = requests.get(f"{SERVER_URL}/tokenizer", timeout=30)
        if t.status_code == 200:
            tok = t.json()
            if "word2id" in tok:
                with open("tokenizer.json","w") as f: json.dump(tok, f)
                print("Tokenizer saved to tokenizer.json")
            else:
                print(f"Tokenizer not ready: {tok.get('error','?')}")
        else:
            print(f"Tokenizer HTTP {t.status_code}")
    except Exception as e:
        print(f"Tokenizer error: {e}")

    print("\nDone!  Run:  python chat.py")

if __name__ == "__main__":
    main()
