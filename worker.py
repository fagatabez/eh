# worker.py — distributed training worker
# Run this to contribute your GPU to the shared training pool.
# It downloads the model & tokenizer from the server, trains on batches,
# and sends gradients back — your local files are never shared.

import os
import uuid
import time
import json
import torch
import requests
import threading
import torch.nn as nn

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

SERVER_URL    = "https://eh-production.up.railway.app"
SHARE_CPU     = False    # set True to donate some CPU to help server
CPU_SHARE_PCT = 10       # % of your CPU to donate (only if SHARE_CPU=True)

# ═══════════════════════════════════════════════════

# ── Inline model definition (so users don't need model.py) ──

import math

class _Config:
    vocab_size  = 5000
    seq_len     = 128
    embed_dim   = 256
    num_heads   = 8
    num_layers  = 6
    dropout     = 0.1

class _SelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.num_heads
        self.d     = cfg.embed_dim // cfg.num_heads
        self.qkv   = nn.Linear(cfg.embed_dim, 3 * cfg.embed_dim)
        self.out   = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.drop  = nn.Dropout(cfg.dropout)
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.heads, self.d).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores  = (q @ k.transpose(-2,-1)) / math.sqrt(self.d)
        mask    = torch.triu(torch.ones(T, T, device=x.device), 1).bool()
        scores  = scores.masked_fill(mask, float('-inf'))
        w       = self.drop(torch.softmax(scores, dim=-1))
        out     = (w @ v).transpose(1,2).reshape(B, T, C)
        return self.out(out)

class _Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn  = _SelfAttention(cfg)
        self.ff    = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim), nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim), nn.Dropout(cfg.dropout)
        )
        self.n1 = nn.LayerNorm(cfg.embed_dim)
        self.n2 = nn.LayerNorm(cfg.embed_dim)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.ff(self.n2(x))
        return x

class _Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = nn.Embedding(cfg.seq_len,    cfg.embed_dim)
        self.blocks  = nn.Sequential(*[_Block(cfg) for _ in range(cfg.num_layers)])
        self.norm    = nn.LayerNorm(cfg.embed_dim)
        self.head    = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        self.drop    = nn.Dropout(cfg.dropout)
    def forward(self, tokens):
        B, T = tokens.shape
        pos  = torch.arange(T, device=tokens.device)
        x    = self.drop(self.tok_emb(tokens) + self.pos_emb(pos))
        x    = self.norm(self.blocks(x))
        return self.head(x)

# ── Helpers ───────────────────────────────────────

def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return 0.0

def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"

def fmt(s):
    if s < 60:   return f"{int(s)}s"
    if s < 3600: return f"{int(s//60)}m{int(s%60)}s"
    return f"{int(s//3600)}h{int((s%3600)//60)}m"

def probe_batch_size(model, device, vram_gb, max_vram_pct, bytes_per_sample):
    """
    Figure out the largest batch that fits in available VRAM.
    Uses the server's bytes_per_sample as the primary guide,
    then does a quick live probe to confirm (and halves if OOM).
    """
    usable_bytes = vram_gb * (1024 ** 3) * max_vram_pct
    estimated    = max(1, int(usable_bytes // bytes_per_sample))
    print(f"  VRAM estimate: {vram_gb:.1f}GB * {max_vram_pct*100:.0f}% / "
          f"{bytes_per_sample}B = {estimated} samples")

    if device.type == "cpu":
        return min(estimated, 8)

    # live OOM probe — try estimated, halve until it works
    model.eval()
    probe = estimated
    while probe >= 1:
        try:
            dummy = torch.zeros(probe, 127, dtype=torch.long, device=device)
            with torch.no_grad():
                model(dummy)
            del dummy
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            probe = probe // 2
            print(f"  OOM at {probe*2} — retrying with {probe}")
    model.train()
    print(f"  Final safe batch size: {probe}")
    return max(1, probe)

# ── Main ──────────────────────────────────────────

def main():
    vram_gb   = get_vram_gb()
    device    = torch.device("cuda" if vram_gb > 0 else "cpu")
    worker_id = str(uuid.uuid4())

    print("=" * 50)
    print(f"  MyAI Worker")
    print(f"  Device : {device} | GPU: {get_gpu_name()}")
    print(f"  VRAM   : {vram_gb:.1f} GB")
    print(f"  CPU    : sharing = {SHARE_CPU} ({CPU_SHARE_PCT}%)")
    print("=" * 50)
    print(f"Connecting to {SERVER_URL} ...")

    # ── Join pool ──────────────────────────────────
    resp = requests.post(f"{SERVER_URL}/join", json={
        "worker_id":    worker_id,
        "vram_gb":      vram_gb,
        "gpu_name":     get_gpu_name(),
        "type":         "script",
        "share_cpu":    SHARE_CPU,
        "cpu_share_pct": CPU_SHARE_PCT if SHARE_CPU else 0,
    }, timeout=10).json()

    server_batch      = resp["batch"]
    bytes_per_sample  = resp.get("bytes_per_sample", 4096)
    max_vram_pct      = resp.get("max_vram_pct",     0.70)
    cap               = resp.get("cap", server_batch)

    print(f"Server assigned : {server_batch} batches (cap={cap})")

    # ── Download model ─────────────────────────────
    print("Downloading model from server...")
    model_bytes = requests.get(f"{SERVER_URL}/model", timeout=60).content
    with open("_worker_model.pt", "wb") as f:
        f.write(model_bytes)

    # ── Download tokenizer ─────────────────────────
    print("Downloading tokenizer...")
    tok_data = requests.get(f"{SERVER_URL}/tokenizer", timeout=30).json()
    word2id  = tok_data["word2id"]
    vocab_size = len(word2id)

    # ── Build model ────────────────────────────────
    checkpoint = torch.load("_worker_model.pt", map_location=device)
    cfg        = _Config()
    for k, v in checkpoint.get("config", {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.vocab_size = vocab_size

    model = _Model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()

    # ── Probe safe batch size ──────────────────────
    print("Probing safe batch size...")
    safe_batch = probe_batch_size(model, device, vram_gb, max_vram_pct, bytes_per_sample)
    # use the smaller of what server assigned and what safely fits
    batch_size = min(server_batch, safe_batch)
    print(f"Using batch size: {batch_size}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # ── Heartbeat ──────────────────────────────────
    def heartbeat():
        while True:
            time.sleep(10)
            try:
                r = requests.post(f"{SERVER_URL}/ping", json={
                    "worker_id":    worker_id,
                    "share_cpu":    SHARE_CPU,
                    "cpu_share_pct": CPU_SHARE_PCT if SHARE_CPU else 0,
                }, timeout=5).json()
                if r.get("status") != "ok":
                    print("[heartbeat] Server lost — retrying...")
                # pick up live config changes (e.g. max_vram_pct)
                new_cfg = r.get("config", {})
                if new_cfg.get("max_vram_pct") != max_vram_pct:
                    print(f"[config] max_vram_pct updated to {new_cfg['max_vram_pct']*100:.0f}%")
            except Exception as e:
                print(f"[heartbeat] {e}")
    threading.Thread(target=heartbeat, daemon=True).start()

    # ── Training loop ──────────────────────────────
    print("Starting training loop — Ctrl+C to stop\n")
    batches_done = 0
    total_loss   = 0.0
    start        = time.time()

    try:
        while True:
            resp = requests.get(f"{SERVER_URL}/get_batch",
                params={"worker_id": worker_id, "size": batch_size},
                timeout=30)

            if resp.status_code == 204:
                print("No batches available — waiting 5s...")
                time.sleep(5)
                continue

            bd     = resp.json()
            tokens = torch.tensor(bd["tokens"], dtype=torch.long).to(device)
            x      = tokens[:, :-1]
            y      = tokens[:, 1:]

            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss   = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss   = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # ── Send gradients ─────────────────────
            grads = {
                name: param.grad.cpu().tolist()
                for name, param in model.named_parameters()
                if param.grad is not None
            }
            requests.post(f"{SERVER_URL}/submit_gradients", json={
                "worker_id": worker_id,
                "loss":      loss.item(),
                "grads":     grads,
                "batch_id":  bd["batch_id"],
            }, timeout=30)

            batches_done += 1
            total_loss   += loss.item()
            avg_loss      = total_loss / batches_done
            elapsed       = time.time() - start
            rate          = batches_done / elapsed if elapsed > 0 else 0

            print(f"[{batches_done:4d}] loss: {loss.item():.4f} | "
                  f"avg: {avg_loss:.4f} | {rate:.2f} batch/s | "
                  f"up: {fmt(elapsed)}")

    except KeyboardInterrupt:
        print("\nStopping — returning batches to pool...")
        try:
            requests.post(f"{SERVER_URL}/leave",
                json={"worker_id": worker_id}, timeout=5)
            print("Done! Thanks for contributing.")
        except:
            pass
    finally:
        if os.path.exists("_worker_model.pt"):
            os.remove("_worker_model.pt")

if __name__ == "__main__":
    main()
