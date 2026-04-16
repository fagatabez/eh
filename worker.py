# worker.py — contribute GPU/CPU to help train MyAI
# ─────────────────────────────────────────────────────────────────────
# How it ACTUALLY helps:
#   1. Downloads the CURRENT trained model weights from the server
#      (train.py pushes updated weights every epoch)
#   2. Downloads a batch of training data from the server
#   3. Runs forward + backward on YOUR hardware
#   4. Sends the gradients (not the model) back to the server
#   5. train.py fetches these gradients and blends them into its own
#      update — effectively processing more data per step than it
#      could alone
#
# This means with 5 workers each processing 50 samples, train.py
# gets updates from 250 extra samples per step it would have missed.
# Workers with more VRAM contribute more samples per batch.
# ─────────────────────────────────────────────────────────────────────

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
        if not _pip(_pkg): print(f"Failed: pip install {_pkg}"); sys.exit(1)
        print(f"{_pkg} OK — restarting..."); os.execv(sys.executable,[sys.executable]+sys.argv)

import uuid, time, math, json, threading, io
import torch, requests
import torch.nn as nn

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

SERVER_URL   = "https://eh-production.up.railway.app"
# How often (in training loops) to refresh model weights from server.
# Lower = workers stay closer to train.py's current weights = better gradients.
# Higher = fewer downloads = less bandwidth.
REFRESH_WEIGHTS_EVERY = 20   # batches between weight refreshes

# ═══════════════════════════════════════════════════
#   INLINE MODEL  (mirrors the architecture in model.py)
# ═══════════════════════════════════════════════════

class _Cfg:
    vocab_size = 5000; seq_len = 128; embed_dim = 256
    num_heads = 8; num_layers = 6; dropout = 0.1

class _Attn(nn.Module):
    def __init__(self,c):
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
        return self.out((self.dp(torch.softmax(s,dim=-1))@v).transpose(1,2).reshape(B,T,C))

class _Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.a=_Attn(c)
        self.ff=nn.Sequential(nn.Linear(c.embed_dim,4*c.embed_dim),nn.GELU(),
                               nn.Linear(4*c.embed_dim,c.embed_dim),nn.Dropout(c.dropout))
        self.n1=nn.LayerNorm(c.embed_dim); self.n2=nn.LayerNorm(c.embed_dim)
    def forward(self,x):
        return x+self.ff(self.n2(x+self.a(self.n1(x))))

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
        return self.head(self.norm(self.blocks(
            self.dp(self.tok_emb(t)+self.pos_emb(torch.arange(T,device=t.device))))))

# ═══════════════════════════════════════════════════
#   HELPERS
# ═══════════════════════════════════════════════════

def get_vram():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory/1024**3
    return 0.0

def get_gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

def fmt(s):
    if s<60: return f"{int(s)}s"
    if s<3600: return f"{int(s//60)}m{int(s%60)}s"
    return f"{int(s//3600)}h{int((s%3600)//60)}m"

def probe_batch(model, device, vram_gb, seq_len, reserve_gb=1.2):
    """Find largest batch that fits with reserve free."""
    if device.type=="cpu": return 4
    usable=vram_gb-reserve_gb
    if usable<=0: return 1
    model.eval()
    probe=max(1,int(usable*200))
    while probe>=1:
        try:
            torch.cuda.empty_cache()
            d=torch.zeros(probe,seq_len-1,dtype=torch.long,device=device)
            with torch.no_grad(): model(d)
            del d; torch.cuda.empty_cache(); break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); probe//=2
    model.train()
    return max(1,probe)

# ═══════════════════════════════════════════════════
#   MODEL MANAGEMENT
#   Workers download the real trained weights so their
#   gradients actually match what train.py needs.
# ═══════════════════════════════════════════════════

_model_etag = None   # track server model version to avoid re-downloading same weights

def download_model_weights(model, device):
    """
    Download the latest model weights from server and load into model.
    Uses ETag to skip download if weights haven't changed.
    Returns True if weights were updated, False if unchanged or failed.
    """
    global _model_etag
    try:
        headers = {}
        if _model_etag: headers["If-None-Match"] = _model_etag
        r = requests.get(f"{SERVER_URL}/model", headers=headers, timeout=30)

        if r.status_code == 304:
            return False  # not modified — skip

        if r.status_code != 200 or len(r.content) < 1000:
            return False

        _model_etag = r.headers.get("ETag")

        # Load weights into the model
        buf  = io.BytesIO(r.content)
        ckpt = torch.load(buf, map_location=device)
        state = ckpt.get("model", ckpt)  # handle both formats
        m = model.module if hasattr(model,"module") else model
        try:
            m.load_state_dict(state, strict=False)
            return True
        except Exception as e:
            print(f"  [weights] load failed: {e}"); return False
    except Exception as e:
        print(f"  [weights] download failed: {e}"); return False

# ═══════════════════════════════════════════════════
#   MAIN
# ═══════════════════════════════════════════════════

def main():
    vram_gb   = get_vram()
    device    = torch.device("cuda" if vram_gb > 0 else "cpu")
    worker_id = str(uuid.uuid4())

    print("="*54)
    print("  MyAI Worker  —  real GPU contribution")
    print(f"  Device : {device} | {get_gpu_name()}")
    print(f"  VRAM   : {vram_gb:.1f} GB")
    print("="*54)
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
    cfg_snap     = resp.get("config", {})
    min_batch    = cfg_snap.get("min_batch", 1)

    print(f"Assigned: {server_batch} batches (cap={cap})")

    # ── Download tokenizer ─────────────────────────
    print("Downloading tokenizer...")
    try:
        tok_r = requests.get(f"{SERVER_URL}/tokenizer", timeout=30)
        if tok_r.status_code != 200:
            print(f"Tokenizer not ready (HTTP {tok_r.status_code}). "
                  "Run train.py first."); return
        tok_data = tok_r.json()
        if "word2id" not in tok_data:
            print(f"Bad tokenizer: {tok_data.get('error','?')}"); return
        vocab_size = len(tok_data["word2id"])
        print(f"Vocab: {vocab_size} tokens")
    except Exception as e:
        print(f"Tokenizer error: {e}"); return

    # ── Check model is available ───────────────────
    print("Checking model on server...")
    try:
        model_check = requests.head(f"{SERVER_URL}/model", timeout=10)
        if model_check.status_code == 404:
            print("No model on server yet. Run train.py first until it syncs "
                  f"(every {cfg_snap.get('sync_every',5)} epochs).")
            print("Waiting 30s then retrying...")
            time.sleep(30)
            model_check = requests.head(f"{SERVER_URL}/model", timeout=10)
            if model_check.status_code == 404:
                print("Still no model. Start train.py first, then run this."); return
    except Exception:
        pass  # HEAD might not be supported, continue

    # ── Build model from architecture ──────────────
    print("Building model...")
    # Try to get architecture from server config
    try:
        srv_cfg = requests.get(f"{SERVER_URL}/config", timeout=10).json()
    except Exception:
        srv_cfg = {}

    wcfg = _Cfg(); wcfg.vocab_size = vocab_size
    for k in ("embed_dim","num_heads","num_layers","seq_len","dropout"):
        if k in srv_cfg: setattr(wcfg, k, srv_cfg[k])

    model = _Model(wcfg).to(device)

    # ── Download REAL trained weights ──────────────
    print("Downloading trained model weights...")
    updated = download_model_weights(model, device)
    if not updated:
        # Try unconditionally
        try:
            r = requests.get(f"{SERVER_URL}/model", timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                buf  = io.BytesIO(r.content)
                ckpt = torch.load(buf, map_location=device)
                state = ckpt.get("model", ckpt)
                m = model.module if hasattr(model,"module") else model
                m.load_state_dict(state, strict=False)
                print("  Weights loaded from server")
            else:
                print("  No model on server — using random init (gradients will be weak)")
                print("  Let train.py run for at least 1 epoch first!")
        except Exception as e:
            print(f"  Weight load failed: {e} — using random init")
    else:
        print("  Weights loaded")
    model.train()

    # ── Probe safe batch size ──────────────────────
    print("Probing safe batch size...")
    safe       = probe_batch(model, device, vram_gb, wcfg.seq_len)
    batch_size = max(min_batch, min(server_batch, safe))
    print(f"Batch size: {batch_size}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
    scaler    = torch.cuda.amp.GradScaler() if device.type=="cuda" else None

    # Live config (updated by heartbeat)
    live = {"batch": batch_size, "min_batch": min_batch}

    # ── Heartbeat ──────────────────────────────────
    def heartbeat():
        while True:
            time.sleep(10)
            try:
                r = requests.post(f"{SERVER_URL}/ping",
                    json={"worker_id": worker_id}, timeout=5).json()
                if r.get("status") == "not_found":
                    # Server restarted — rejoin
                    rj = requests.post(f"{SERVER_URL}/join", json={
                        "worker_id": worker_id, "vram_gb": vram_gb,
                        "gpu_name": get_gpu_name(), "type": "script",
                    }, timeout=10).json()
                    live["batch"] = max(live["min_batch"],
                        min(rj.get("batch", live["batch"]), safe))
                    print(f"[reconnected] batch={live['batch']}")
                elif r.get("status") == "ok":
                    nc = r.get("config", {})
                    live["min_batch"] = nc.get("min_batch", 1)
                    live["batch"] = max(live["min_batch"],
                        min(r.get("batch", live["batch"]), safe))
            except Exception as e:
                print(f"[heartbeat] {e}")
    threading.Thread(target=heartbeat, daemon=True).start()

    # ── Training loop ──────────────────────────────
    print("Contributing computing power — press Ctrl+C to stop\n")
    done = 0; total_loss = 0.0; t0 = time.time()
    last_weight_refresh = 0

    try:
        while True:
            bs = live["batch"]

            # ── Refresh model weights from server ─────
            # Critical: without this, worker gradients point in the wrong
            # direction (based on old weights) and hurt training instead of helping.
            if done - last_weight_refresh >= REFRESH_WEIGHTS_EVERY:
                updated = download_model_weights(model, device)
                if updated:
                    # Reset optimizer momentum since weights changed significantly
                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
                    last_weight_refresh = done
                    print(f"  [weights] refreshed from server (batch {done})")

            # ── Get training batch from server ────────
            try:
                resp = requests.get(f"{SERVER_URL}/get_batch",
                    params={"worker_id": worker_id, "size": bs}, timeout=30)
            except Exception as e:
                print(f"Batch error: {e}"); time.sleep(5); continue

            if resp.status_code == 204:
                print("No data — waiting..."); time.sleep(5); continue
            if resp.status_code != 200:
                time.sleep(5); continue

            bd     = resp.json()
            tokens = torch.tensor(bd["tokens"], dtype=torch.long).to(device)
            x, y   = tokens[:, :-1], tokens[:, 1:]

            # ── Forward + backward ────────────────────
            optimizer.zero_grad()
            try:
                if scaler:
                    with torch.cuda.amp.autocast():
                        logits = model(x)
                        loss   = loss_fn(logits.reshape(-1, wcfg.vocab_size), y.reshape(-1))
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    logits = model(x)
                    loss   = loss_fn(logits.reshape(-1, wcfg.vocab_size), y.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache(); optimizer.zero_grad()
                live["batch"] = max(1, live["batch"]//2)
                print(f"OOM — batch reduced to {live['batch']}"); continue

            # ── Send gradients to server ───────────────
            # Only send gradients for parameters that actually changed.
            # Compress: send mean+std instead of full tensor for large params.
            grads = {}
            for name, param in model.named_parameters():
                if param.grad is None: continue
                g = param.grad.cpu()
                # For large params (>10k elements), quantize to save bandwidth
                if g.numel() > 10_000:
                    grads[name] = g.half().tolist()   # float16 = 2x smaller
                else:
                    grads[name] = g.tolist()

            try:
                requests.post(f"{SERVER_URL}/submit_gradients", json={
                    "worker_id": worker_id,
                    "loss":      loss.item(),
                    "grads":     grads,
                    "batch_id":  bd["batch_id"],
                }, timeout=30)
            except Exception:
                pass   # gradient loss is recoverable

            done += 1; total_loss += loss.item(); avg = total_loss/done
            ela = time.time()-t0; rate = done/ela if ela>0 else 0
            print(f"[{done:4d}] loss:{loss.item():.4f} avg:{avg:.4f} "
                  f"batch:{bs} {rate:.2f}b/s up:{fmt(ela)}")

    except KeyboardInterrupt:
        print("\nStopping — returning batches to pool...")
        try:
            requests.post(f"{SERVER_URL}/leave",
                json={"worker_id": worker_id}, timeout=5)
            print(f"Done! Contributed {done} batches. Thanks!")
        except Exception: pass

if __name__ == "__main__":
    main()
