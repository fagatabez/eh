# worker.py — contribute your GPU/CPU to help train MyAI
# ─────────────────────────────────────────────────────────
# What this does:
#   1. Downloads PUBLIC training data batches from the server
#   2. Trains a local copy of the model on those batches
#   3. Sends only the GRADIENTS back — your trained model stays private
#      and the server never gets your personal myai.pt file
#
# The server provides:
#   - Batches of training text (public data you helped download)
#   - The model architecture config (no weights, just shape)
#   - Tokenizer vocab
#   - Admin settings (CPU/GPU share limits set by the owner)
#
# Your GPU/CPU does real training work that helps improve the AI.
# ─────────────────────────────────────────────────────────

# Auto-install missing packages
import sys, subprocess, os

def _pip(pkg):
    for flags in [["--break-system-packages","-q"],["-q"]]:
        r = subprocess.run([sys.executable,"-m","pip","install",pkg]+flags,
                           capture_output=True,text=True)
        if r.returncode==0: return True
    return False

for _pkg in ["torch","requests"]:
    try: __import__(_pkg)
    except ImportError:
        print(f"Installing {_pkg}...")
        if not _pip(_pkg):
            print(f"Failed to install {_pkg}. Run: pip install {_pkg}"); sys.exit(1)
        print(f"{_pkg} installed — restarting...")
        os.execv(sys.executable, [sys.executable]+sys.argv)

import uuid, time, math, json, threading
import torch, requests
import torch.nn as nn

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

SERVER_URL = "https://eh-production.up.railway.app"

# These are overridden by admin settings from the server
SHARE_CPU     = False   # donate CPU time even without a GPU
CPU_THREADS   = 2       # CPU threads to use (server can override)

# ═══════════════════════════════════════════════════
#   INLINE MODEL  (no model.py needed on worker machine)
# ═══════════════════════════════════════════════════

class _Cfg:
    vocab_size = 5000; seq_len = 128; embed_dim = 256
    num_heads = 8; num_layers = 6; dropout = 0.1

class _Attn(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h=c.num_heads; self.d=c.embed_dim//c.num_heads
        self.qkv=nn.Linear(c.embed_dim,3*c.embed_dim)
        self.out=nn.Linear(c.embed_dim,c.embed_dim)
        self.dp=nn.Dropout(c.dropout)
    def forward(self,x):
        B,T,C=x.shape
        qkv=self.qkv(x).reshape(B,T,3,self.h,self.d).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        s=(q@k.transpose(-2,-1))/math.sqrt(self.d)
        mask=torch.triu(torch.ones(T,T,device=x.device),1).bool()
        s=s.masked_fill(mask,float("-inf"))
        w=self.dp(torch.softmax(s,dim=-1))
        return self.out((w@v).transpose(1,2).reshape(B,T,C))

class _Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.a=_Attn(c)
        self.ff=nn.Sequential(nn.Linear(c.embed_dim,4*c.embed_dim),nn.GELU(),
                               nn.Linear(4*c.embed_dim,c.embed_dim),nn.Dropout(c.dropout))
        self.n1=nn.LayerNorm(c.embed_dim); self.n2=nn.LayerNorm(c.embed_dim)
    def forward(self,x):
        x=x+self.a(self.n1(x)); return x+self.ff(self.n2(x))

class _Model(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.tok_emb=nn.Embedding(c.vocab_size,c.embed_dim)
        self.pos_emb=nn.Embedding(c.seq_len,c.embed_dim)
        self.blocks=nn.Sequential(*[_Block(c) for _ in range(c.num_layers)])
        self.norm=nn.LayerNorm(c.embed_dim)
        self.head=nn.Linear(c.embed_dim,c.vocab_size,bias=False)
        self.dp=nn.Dropout(c.dropout); self.cfg=c
    def forward(self,t):
        B,T=t.shape
        x=self.dp(self.tok_emb(t)+self.pos_emb(torch.arange(T,device=t.device)))
        return self.head(self.norm(self.blocks(x)))

# ═══════════════════════════════════════════════════
#   HELPERS
# ═══════════════════════════════════════════════════

def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory/1024**3
    return 0.0

def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def fmt(s):
    if s<60:   return f"{int(s)}s"
    if s<3600: return f"{int(s//60)}m{int(s%60)}s"
    return f"{int(s//3600)}h{int((s%3600)//60)}m"

def probe_batch(model, device, vram_gb, seq_len, reserve_gb=1.0):
    """Find largest batch that fits with reserve_gb free."""
    if device.type == "cpu":
        return 4
    usable = vram_gb - reserve_gb
    if usable <= 0: return 1
    model.eval()
    probe = max(1, int(usable * 256))
    while probe >= 1:
        try:
            torch.cuda.empty_cache()
            d = torch.zeros(probe, seq_len-1, dtype=torch.long, device=device)
            with torch.no_grad(): model(d)
            del d; torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); probe //= 2
    model.train()
    return max(1, probe)

# ═══════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════

def main():
    vram_gb   = get_vram_gb()
    device    = torch.device("cuda" if vram_gb > 0 else "cpu")
    worker_id = str(uuid.uuid4())

    print("="*50)
    print("  MyAI Worker  —  sharing computing power")
    print(f"  Device : {device} | {get_gpu_name()}")
    print(f"  VRAM   : {vram_gb:.1f} GB")
    print("="*50)
    print(f"Connecting to {SERVER_URL} ...")

    # ── Join pool ──────────────────────────────────
    try:
        resp = requests.post(f"{SERVER_URL}/join", json={
            "worker_id": worker_id, "vram_gb": vram_gb,
            "gpu_name": get_gpu_name(), "type": "script",
        }, timeout=10).json()
    except Exception as e:
        print(f"Cannot connect: {e}"); return

    server_batch = resp.get("batch", 4)
    cap          = resp.get("cap", server_batch)

    # ── Apply admin config ─────────────────────────
    cfg_snap         = resp.get("config", {})
    bytes_per_sample = cfg_snap.get("bytes_per_sample", 32_212_254)
    max_vram_pct     = cfg_snap.get("max_vram_pct", 0.70)
    min_batch        = cfg_snap.get("min_batch", 1)
    # CPU sharing: admin can allow workers to donate CPU
    allow_cpu_share  = cfg_snap.get("allow_cpu_share", False)
    cpu_threads_cfg  = cfg_snap.get("cpu_threads", CPU_THREADS)

    print(f"Admin config: allow_cpu={allow_cpu_share} threads={cpu_threads_cfg} "
          f"max_vram={max_vram_pct*100:.0f}%")
    print(f"Assigned: {server_batch} batches (cap={cap})")

    # Apply CPU thread limit if sharing is enabled
    if allow_cpu_share or device.type == "cpu":
        torch.set_num_threads(cpu_threads_cfg)
        print(f"CPU threads: {cpu_threads_cfg}")

    # ── Download tokenizer vocab ───────────────────
    # We only need the vocab to build the model — not the model weights.
    # This is PUBLIC data (the tokenizer is built from public text).
    print("\nDownloading tokenizer vocab...")
    try:
        tok_r = requests.get(f"{SERVER_URL}/tokenizer", timeout=30)
        if tok_r.status_code != 200:
            print(f"Tokenizer not ready (HTTP {tok_r.status_code}).")
            print("The server owner needs to run train.py once to upload it.")
            return
        tok_data = tok_r.json()
        if "error" in tok_data or "word2id" not in tok_data:
            print(f"Tokenizer error: {tok_data.get('error','bad format')}")
            print("Run train.py first on the server owner's machine.")
            return
        word2id    = tok_data["word2id"]
        vocab_size = len(word2id)
        print(f"Vocab loaded: {vocab_size} tokens")
    except Exception as e:
        print(f"Tokenizer download failed: {e}"); return

    # ── Download model CONFIG (not weights) ────────
    # We build a fresh model from scratch using the same architecture.
    # Workers train from a random init on public data — this is fine
    # because we're only sending GRADIENTS (direction of improvement),
    # not the model itself.  The server owner's weights stay private.
    print("Building local model from architecture config...")
    try:
        # Try to get architecture config from server
        cfg_r = requests.get(f"{SERVER_URL}/config", timeout=10)
        server_cfg = cfg_r.json() if cfg_r.status_code == 200 else {}
    except Exception:
        server_cfg = {}

    wcfg = _Cfg()
    wcfg.vocab_size = vocab_size
    # Apply any architecture hints from server config
    for k in ("embed_dim", "num_heads", "num_layers", "seq_len", "dropout"):
        if k in server_cfg:
            setattr(wcfg, k, server_cfg[k])

    model = _Model(wcfg).to(device)
    model.train()

    # ── Probe safe batch size ──────────────────────
    print("Probing safe batch size...")
    reserve = 1.0 if device.type == "cuda" else 0
    safe    = probe_batch(model, device, vram_gb, wcfg.seq_len, reserve_gb=reserve)
    batch_size = max(min_batch, min(server_batch, safe))
    print(f"Batch size: {batch_size}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
    scaler    = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Live config (updated by heartbeat)
    live = {"batch_size": batch_size, "bytes_per_sample": bytes_per_sample,
            "max_vram_pct": max_vram_pct, "min_batch": min_batch,
            "allow_cpu_share": allow_cpu_share, "cpu_threads": cpu_threads_cfg}

    # ── Heartbeat thread ───────────────────────────
    def heartbeat():
        while True:
            time.sleep(10)
            try:
                r = requests.post(f"{SERVER_URL}/ping",
                    json={"worker_id": worker_id}, timeout=5).json()
                if r.get("status") != "ok": continue
                nc = r.get("config", {})
                changed = False
                for key in ("bytes_per_sample","max_vram_pct","min_batch"):
                    if nc.get(key) != live.get(key):
                        live[key] = nc[key]; changed = True
                # CPU sharing
                if nc.get("allow_cpu_share") != live.get("allow_cpu_share"):
                    live["allow_cpu_share"] = nc.get("allow_cpu_share", False)
                    live["cpu_threads"]     = nc.get("cpu_threads", CPU_THREADS)
                    if live["allow_cpu_share"] or device.type == "cpu":
                        torch.set_num_threads(live["cpu_threads"])
                    changed = True
                if changed:
                    new_safe = probe_batch(model, device, vram_gb,
                                           wcfg.seq_len, reserve_gb=reserve)
                    live["batch_size"] = max(live["min_batch"],
                        min(r.get("batch", server_batch), new_safe))
                    print(f"[config] updated batch={live['batch_size']}")
            except Exception as e:
                print(f"[heartbeat] {e}")
    threading.Thread(target=heartbeat, daemon=True).start()

    # ── Training loop ──────────────────────────────
    print("Training — press Ctrl+C to stop and return batches\n")
    done = 0; total_loss = 0.0; t0 = time.time()

    try:
        while True:
            bs = live["batch_size"]
            try:
                resp = requests.get(f"{SERVER_URL}/get_batch",
                    params={"worker_id": worker_id, "size": bs}, timeout=30)
            except Exception as e:
                print(f"Batch fetch error: {e} — retrying in 5s")
                time.sleep(5); continue

            if resp.status_code == 204:
                print("No data available — waiting..."); time.sleep(5); continue
            if resp.status_code != 200:
                time.sleep(5); continue

            bd     = resp.json()
            tokens = torch.tensor(bd["tokens"], dtype=torch.long).to(device)
            x, y   = tokens[:, :-1], tokens[:, 1:]

            optimizer.zero_grad()
            try:
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        loss = loss_fn(logits.reshape(-1, wcfg.vocab_size), y.reshape(-1))
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    logits = model(x)
                    loss = loss_fn(logits.reshape(-1, wcfg.vocab_size), y.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); optimizer.zero_grad()
                live["batch_size"] = max(1, live["batch_size"] // 2)
                print(f"OOM — reduced batch to {live['batch_size']}"); continue

            # ── Send gradients (not weights) ───────
            grads = {name: param.grad.cpu().tolist()
                     for name, param in model.named_parameters()
                     if param.grad is not None}
            try:
                requests.post(f"{SERVER_URL}/submit_gradients", json={
                    "worker_id": worker_id, "loss": loss.item(),
                    "grads": grads, "batch_id": bd["batch_id"],
                }, timeout=30)
            except Exception:
                pass  # gradient loss is recoverable

            done += 1; total_loss += loss.item()
            avg  = total_loss / done; ela = time.time()-t0
            rate = done/ela if ela > 0 else 0
            print(f"[{done:4d}] loss:{loss.item():.4f} avg:{avg:.4f} "
                  f"{rate:.2f}b/s up:{fmt(ela)}")

    except KeyboardInterrupt:
        print("\nStopping — returning batches...")
        try:
            requests.post(f"{SERVER_URL}/leave",
                json={"worker_id": worker_id}, timeout=5)
            print("Done! Thanks for contributing.")
        except Exception:
            pass

if __name__ == "__main__":
    main()
