# worker.py — distributed training worker
# Run on any machine to donate GPU/CPU to the shared training pool.
# Downloads model + tokenizer from server, trains on batches,
# sends gradients back. Your local files are never shared.

import os
import uuid
import time
import json
import math
import torch
import requests
import threading
import torch.nn as nn

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

SERVER_URL    = "https://eh-production.up.railway.app"
SHARE_CPU     = False
CPU_SHARE_PCT = 10

# ═══════════════════════════════════════════════════
#   INLINE MODEL (no model.py needed)
# ═══════════════════════════════════════════════════

class _Config:
    vocab_size = 5000
    seq_len    = 128
    embed_dim  = 256
    num_heads  = 8
    num_layers = 6
    dropout    = 0.1

class _Attn(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h   = cfg.num_heads
        self.d   = cfg.embed_dim // cfg.num_heads
        self.qkv = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.out = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dp  = nn.Dropout(cfg.dropout)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.d).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        s = (q @ k.transpose(-2,-1)) / math.sqrt(self.d)
        mask = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        s = s.masked_fill(mask, float("-inf"))
        w = self.dp(torch.softmax(s, dim=-1))
        return self.out((w @ v).transpose(1,2).reshape(B, T, C))

class _Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.a  = _Attn(cfg)
        self.ff = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4*cfg.embed_dim), nn.GELU(),
            nn.Linear(4*cfg.embed_dim, cfg.embed_dim), nn.Dropout(cfg.dropout)
        )
        self.n1 = nn.LayerNorm(cfg.embed_dim)
        self.n2 = nn.LayerNorm(cfg.embed_dim)
    def forward(self, x):
        x = x + self.a(self.n1(x))
        return x + self.ff(self.n2(x))

class _Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.embed_dim)
        self.blocks  = nn.Sequential(*[_Block(cfg) for _ in range(cfg.num_layers)])
        self.norm    = nn.LayerNorm(cfg.embed_dim)
        self.head    = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.dp      = nn.Dropout(cfg.dropout)
        self.cfg     = cfg
    def forward(self, t):
        B, T = t.shape
        x = self.dp(self.tok_emb(t) + self.pos_emb(torch.arange(T, device=t.device)))
        return self.head(self.norm(self.blocks(x)))

# ═══════════════════════════════════════════════════
#   HELPERS
# ═══════════════════════════════════════════════════

def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0.0

def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def fmt(s):
    if s < 60:   return f"{int(s)}s"
    if s < 3600: return f"{int(s//60)}m{int(s%60)}s"
    return f"{int(s//3600)}h{int((s%3600)//60)}m"

def probe_batch(model, device, vram_gb, max_vram_pct, bytes_per_sample):
    usable   = vram_gb * (1024**3) * max_vram_pct
    estimate = max(1, int(usable // bytes_per_sample))
    print(f"  VRAM estimate: {vram_gb:.1f}GB × {max_vram_pct*100:.0f}% / "
          f"{bytes_per_sample//1024//1024}MB = {estimate} samples")
    if device.type == "cpu":
        return min(estimate, 8)
    model.eval()
    probe = estimate
    while probe >= 1:
        try:
            dummy = torch.zeros(probe, 127, dtype=torch.long, device=device)
            with torch.no_grad():
                model(dummy)
            del dummy; torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            probe = probe // 2
            print(f"  OOM — retrying with {probe}")
    model.train()
    return max(1, probe)

# ═══════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════

def main():
    vram_gb   = get_vram_gb()
    device    = torch.device("cuda" if vram_gb > 0 else "cpu")
    worker_id = str(uuid.uuid4())

    print("=" * 50)
    print("  MyAI Worker")
    print(f"  Device : {device} | GPU: {get_gpu_name()}")
    print(f"  VRAM   : {vram_gb:.1f} GB")
    print("=" * 50)
    print(f"Connecting to {SERVER_URL} ...")

    # ── Join pool ──────────────────────────────────
    try:
        resp = requests.post(f"{SERVER_URL}/join", json={
            "worker_id":     worker_id,
            "vram_gb":       vram_gb,
            "gpu_name":      get_gpu_name(),
            "type":          "script",
            "share_cpu":     SHARE_CPU,
            "cpu_share_pct": CPU_SHARE_PCT if SHARE_CPU else 0,
        }, timeout=10).json()
    except Exception as e:
        print(f"Could not connect to server: {e}")
        return

    server_batch = resp["batch"]
    cap          = resp.get("cap", server_batch)

    # ── Apply admin config from server ─────────────
    # Server always sends the latest admin settings on /join and /ping
    cfg_snap         = resp.get("config", {})
    bytes_per_sample = cfg_snap.get("bytes_per_sample", 32_212_254)
    max_vram_pct     = cfg_snap.get("max_vram_pct", 0.70)
    min_batch        = cfg_snap.get("min_batch", 1)
    print(f"Admin config: bytes/sample={bytes_per_sample//1024//1024}MB "
          f"max_vram={max_vram_pct*100:.0f}% min_batch={min_batch}")
    print(f"Server assigned: {server_batch} batches (cap={cap})")

    # ── Download model ─────────────────────────────
    print("Downloading model...")
    try:
        model_resp = requests.get(f"{SERVER_URL}/model", timeout=60)
        if model_resp.status_code != 200:
            print("No model on server yet — try again after train.py has run once.")
            return
        with open("_worker_model.pt", "wb") as f:
            f.write(model_resp.content)
    except Exception as e:
        print(f"Model download failed: {e}")
        return

    # ── Download tokenizer ─────────────────────────
    print("Downloading tokenizer...")
    try:
        tok_resp = requests.get(f"{SERVER_URL}/tokenizer", timeout=30)
        if tok_resp.status_code != 200:
            print(f"Tokenizer not available yet (HTTP {tok_resp.status_code}).")
            print("Run train.py at least once so it uploads the tokenizer, then retry.")
            return
        tok_data   = tok_resp.json()
        # Guard against server returning an error JSON like {"error": "..."}
        if "error" in tok_data or "word2id" not in tok_data:
            print(f"Server tokenizer error: {tok_data.get('error', tok_data)}")
            print("Run train.py first so it uploads the tokenizer.")
            return
        word2id    = tok_data["word2id"]
        vocab_size = len(word2id)
        print(f"Tokenizer loaded: {vocab_size} tokens")
    except Exception as e:
        print(f"Tokenizer download failed: {e}")
        return

    # ── Build model ────────────────────────────────
    checkpoint = torch.load("_worker_model.pt", map_location=device)
    wcfg       = _Config()
    for k, v in checkpoint.get("config", {}).items():
        if hasattr(wcfg, k):
            setattr(wcfg, k, v)
    wcfg.vocab_size = vocab_size

    model = _Model(wcfg).to(device)
    try:
        model.load_state_dict(checkpoint["model"])
    except Exception as e:
        print(f"Model load error: {e}")
        return
    model.train()

    # ── Probe safe batch size ──────────────────────
    print("Probing safe batch size...")
    safe_batch = probe_batch(model, device, vram_gb, max_vram_pct, bytes_per_sample)
    batch_size = max(min_batch, min(server_batch, safe_batch))
    print(f"Using batch size: {batch_size}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Heartbeat: ping + update local config ──────
    def heartbeat():
        nonlocal batch_size, bytes_per_sample, max_vram_pct, min_batch
        while True:
            time.sleep(10)
            try:
                r = requests.post(f"{SERVER_URL}/ping", json={
                    "worker_id": worker_id,
                }, timeout=5).json()
                if r.get("status") != "ok":
                    continue
                # ── Apply any admin config changes ──
                new_cfg = r.get("config", {})
                changed = False
                if new_cfg.get("max_vram_pct") != max_vram_pct:
                    max_vram_pct = new_cfg["max_vram_pct"]
                    changed = True
                if new_cfg.get("bytes_per_sample") != bytes_per_sample:
                    bytes_per_sample = new_cfg["bytes_per_sample"]
                    changed = True
                if new_cfg.get("min_batch", min_batch) != min_batch:
                    min_batch = new_cfg["min_batch"]
                    changed = True
                if changed:
                    new_safe = probe_batch(model, device, vram_gb, max_vram_pct, bytes_per_sample)
                    new_srv  = r.get("batch", server_batch)
                    batch_size = max(min_batch, min(new_srv, new_safe))
                    print(f"[config update] new batch_size={batch_size} "
                          f"max_vram={max_vram_pct*100:.0f}% "
                          f"bytes/sample={bytes_per_sample//1024//1024}MB")
            except Exception as e:
                print(f"[heartbeat] {e}")
    threading.Thread(target=heartbeat, daemon=True).start()

    # ── Training loop ──────────────────────────────
    print("Training — Ctrl+C to stop\n")
    done       = 0
    total_loss = 0.0
    t0         = time.time()

    try:
        while True:
            try:
                resp = requests.get(
                    f"{SERVER_URL}/get_batch",
                    params={"worker_id": worker_id, "size": batch_size},
                    timeout=30)
            except Exception as e:
                print(f"Batch fetch error: {e} — retrying in 5s")
                time.sleep(5)
                continue

            if resp.status_code == 204:
                print("No batches — waiting 5s...")
                time.sleep(5)
                continue
            if resp.status_code != 200:
                print(f"Server error {resp.status_code} — retrying in 5s")
                time.sleep(5)
                continue

            bd     = resp.json()
            tokens = torch.tensor(bd["tokens"], dtype=torch.long).to(device)
            x, y   = tokens[:, :-1], tokens[:, 1:]

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss   = loss_fn(logits.reshape(-1, wcfg.vocab_size), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss   = loss_fn(logits.reshape(-1, wcfg.vocab_size), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # ── Send gradients ─────────────────────
            grads = {
                name: param.grad.cpu().tolist()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            try:
                requests.post(f"{SERVER_URL}/submit_gradients", json={
                    "worker_id": worker_id,
                    "loss":      loss.item(),
                    "grads":     grads,
                    "batch_id":  bd["batch_id"],
                }, timeout=30)
            except Exception:
                pass   # gradient loss is recoverable

            done       += 1
            total_loss += loss.item()
            avg         = total_loss / done
            ela         = time.time() - t0
            rate        = done / ela if ela > 0 else 0

            print(f"[{done:4d}] loss:{loss.item():.4f} avg:{avg:.4f} "
                  f"{rate:.2f}b/s up:{fmt(ela)}")

    except KeyboardInterrupt:
        print("\nStopping — returning batches to pool...")
        try:
            requests.post(f"{SERVER_URL}/leave", json={"worker_id": worker_id}, timeout=5)
            print("Done! Thanks for contributing.")
        except Exception:
            pass
    finally:
        try:
            os.remove("_worker_model.pt")
        except Exception:
            pass

if __name__ == "__main__":
    main()
