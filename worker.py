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
    # These are FALLBACK defaults only — the real values are always
    # loaded from the server checkpoint before any training starts.
    vocab_size = 5000; seq_len = 128; embed_dim = 512
    num_heads  = 8;    num_layers = 8; dropout = 0.1

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
        # Names MUST match model.py exactly — attn/norm1/norm2
        # (old worker used a/n1/n2 which caused weight load failure)
        self.attn=_Attn(c)
        self.ff=nn.Sequential(nn.Linear(c.embed_dim,4*c.embed_dim),nn.GELU(),
                               nn.Linear(4*c.embed_dim,c.embed_dim),nn.Dropout(c.dropout))
        self.norm1=nn.LayerNorm(c.embed_dim); self.norm2=nn.LayerNorm(c.embed_dim)
    def forward(self,x):
        return x+self.ff(self.norm2(x+self.attn(self.norm1(x))))

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

def _wmic_gpu_name():
    """Read GPU name from Windows WMI (works for AMD/Intel iGPU)."""
    if os.name != 'nt':
        return None
    try:
        r = subprocess.run(
            'wmic path win32_VideoController get Name /format:value',
            capture_output=True, text=True, shell=True, timeout=5)
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("Name=") and line[5:].strip():
                return line[5:].strip()
    except Exception:
        pass
    return None

def _try_directml():
    """
    Try to get a DirectML device (AMD / Intel iGPU on Windows).
    Returns (device, name, vram_gb) or (None, None, 0).
    Install with:  pip install torch-directml
    """
    try:
        import torch_directml as dml
        dev  = dml.device()
        # Try to get name; some versions expose device_name()
        name = dml.device_name(dml.default_device()) if hasattr(dml, 'device_name') else "DirectML GPU"
        # Shared memory: use psutil to estimate free RAM as usable "VRAM"
        try:
            import psutil
            free_gb = psutil.virtual_memory().available / 1024**3
            vram_gb = round(min(free_gb * 0.7, 8.0), 1)  # use up to 70% of free RAM
        except ImportError:
            vram_gb = 2.0   # safe default
        return dev, name, vram_gb
    except (ImportError, Exception):
        return None, None, 0.0

def get_vram():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    dml_dev, _, dml_vram = _try_directml()
    if dml_dev is not None:
        return dml_vram
    return 0.0

def get_gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    _, dml_name, dml_vram = _try_directml()
    if dml_name:
        return dml_name
    # Last resort: wmic (shows AMD/Intel iGPU even without driver support)
    wmic_name = _wmic_gpu_name()
    return wmic_name or "CPU"

def get_device_and_info():
    """
    Returns (device, gpu_name, vram_gb, device_label).
    Priority: NVIDIA CUDA > AMD/Intel DirectML > CPU
    """
    if torch.cuda.is_available():
        name   = torch.cuda.get_device_name(0)
        vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return torch.device("cuda"), name, vram, "cuda"

    dml_dev, dml_name, dml_vram = _try_directml()
    if dml_dev is not None:
        return dml_dev, dml_name, dml_vram, "directml"

    wmic_name = _wmic_gpu_name() or "CPU"
    return torch.device("cpu"), wmic_name, 0.0, "cpu"

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
    device, gpu_name, vram_gb, device_label = get_device_and_info()
    worker_id = str(uuid.uuid4())

    print("="*54)
    print("  MyAI Worker  —  real GPU contribution")
    print(f"  Device : {device_label} | {gpu_name}")
    print(f"  VRAM   : {vram_gb:.1f} GB" + (
        "  [shared/estimated]" if device_label == "directml" else
        "  [system RAM used]"  if device_label == "cpu"       else ""))
    if device_label == "directml":
        print("  ✓ AMD/Intel iGPU via DirectML")
    elif device_label == "cpu" and gpu_name not in ("CPU", ""):
        print(f"  ⚠ GPU detected ({gpu_name}) but no driver support.")
        print("    Install DirectML for GPU acceleration:")
        print("      pip install torch-directml")
    print("="*54)
    print(f"Connecting to {SERVER_URL} ...")

    # ── Join pool ──────────────────────────────────
    try:
        resp = requests.post(f"{SERVER_URL}/join", json={
            "worker_id": worker_id, "vram_gb": vram_gb,
            "gpu_name": gpu_name, "type": "script",
        }, timeout=10).json()
    except Exception as e:
        print(f"Cannot connect: {e}"); return

    server_batch = resp.get("batch", 4)
    cap          = resp.get("cap", server_batch)
    cfg_snap     = resp.get("config", {})
    min_batch    = cfg_snap.get("min_batch", 1)
    # Read server CPU-share requirement
    server_wants_cpu = cfg_snap.get("allow_cpu_share", False)
    server_cpu_threads = cfg_snap.get("cpu_threads", 2)

    print(f"Assigned: {server_batch} batches (cap={cap})")
    if server_wants_cpu:
        print(f"  Server requests CPU sharing: {server_cpu_threads} thread(s)")
        torch.set_num_threads(server_cpu_threads)
        print(f"  CPU sharing enabled — torch threads set to {server_cpu_threads}")

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
    # Always read architecture from the server checkpoint first.
    # This guarantees the worker matches train.py exactly, regardless
    # of what _Cfg defaults say.
    print("Building model...")

    wcfg = _Cfg(); wcfg.vocab_size = vocab_size

    # Step 1: try to read architecture from the live checkpoint on server
    ckpt_arch_loaded = False
    try:
        r_ckpt = requests.get(f"{SERVER_URL}/model", timeout=60)
        if r_ckpt.status_code == 200 and len(r_ckpt.content) > 1000:
            buf  = io.BytesIO(r_ckpt.content)
            ckpt = torch.load(buf, map_location="cpu")
            cfg_dict = ckpt.get("config", {})
            for k in ("embed_dim","num_heads","num_layers","seq_len","dropout","vocab_size"):
                if k in cfg_dict:
                    setattr(wcfg, k, cfg_dict[k])
            print(f"  Architecture from checkpoint: embed={wcfg.embed_dim} "
                  f"layers={wcfg.num_layers} heads={wcfg.num_heads} vocab={wcfg.vocab_size}")
            ckpt_arch_loaded = True
        else:
            print("  No checkpoint on server — using default architecture")
    except Exception as e:
        print(f"  Could not read checkpoint architecture: {e}")

    # Step 2: fallback — patch from /config if checkpoint wasn't available
    if not ckpt_arch_loaded:
        try:
            srv_cfg = requests.get(f"{SERVER_URL}/config", timeout=10).json()
            for k in ("embed_dim","num_heads","num_layers","seq_len","dropout"):
                if k in srv_cfg: setattr(wcfg, k, srv_cfg[k])
        except Exception:
            pass

    model = _Model(wcfg).to(device)

    # ── Load weights into the model we just built ──
    print("Loading trained model weights...")
    if ckpt_arch_loaded:
        # We already have the checkpoint bytes — load directly
        try:
            buf2  = io.BytesIO(r_ckpt.content)
            ckpt2 = torch.load(buf2, map_location=device)
            state = ckpt2.get("model", ckpt2)
            m = model.module if hasattr(model, "module") else model
            m.load_state_dict(state, strict=True)
            print("  Weights loaded from server checkpoint")
        except Exception as e:
            print(f"  Weight load failed ({e}) — using random init (gradients will be weak)")
    else:
        print("  No model on server — using random init")
        print("  Let train.py run for at least 1 epoch first!")
    model.train()

    # ── Probe safe batch size ──────────────────────
    print("Probing safe batch size...")
    safe       = probe_batch(model, device, vram_gb, wcfg.seq_len)
    batch_size = max(min_batch, min(server_batch, safe))
    print(f"Batch size: {batch_size}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)
    # AMP only works on NVIDIA CUDA — not DirectML or CPU
    scaler    = torch.cuda.amp.GradScaler() if device_label == "cuda" else None

    # Live config (updated by heartbeat)
    live = {"batch": batch_size, "min_batch": min_batch,
            "cpu_share": server_wants_cpu, "cpu_threads": server_cpu_threads}

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
                        "gpu_name": gpu_name, "type": "script",
                    }, timeout=10).json()
                    live["batch"] = max(live["min_batch"],
                        min(rj.get("batch", live["batch"]), safe))
                    print(f"[reconnected] batch={live['batch']}")
                elif r.get("status") == "ok":
                    nc = r.get("config", {})
                    live["min_batch"]   = nc.get("min_batch", 1)
                    live["batch"]       = max(live["min_batch"],
                        min(r.get("batch", live["batch"]), safe))
                    # Live CPU share update
                    new_cpu_share   = nc.get("allow_cpu_share", False)
                    new_cpu_threads = nc.get("cpu_threads", 2)
                    if new_cpu_share != live["cpu_share"] or new_cpu_threads != live["cpu_threads"]:
                        live["cpu_share"]   = new_cpu_share
                        live["cpu_threads"] = new_cpu_threads
                        if new_cpu_share:
                            torch.set_num_threads(new_cpu_threads)
                            print(f"[config] CPU sharing enabled — {new_cpu_threads} thread(s)")
                        else:
                            torch.set_num_threads(max(1, os.cpu_count() or 1))
                            print(f"[config] CPU sharing disabled — threads restored")
            except Exception as e:
                print(f"[heartbeat] {e}")
    threading.Thread(target=heartbeat, daemon=True).start()

    # ── Training loop ──────────────────────────────
    print("="*54)
    print("  Contributing to your AI — Ctrl+C to stop")
    print(f"  Every batch you run helps reduce the training loss.")
    print(f"  Your gradients are blended into train.py in real-time.")
    print("="*54 + "\n")
    done = 0; total_loss = 0.0; t0 = time.time()
    last_weight_refresh = 0
    _no_data_warned = False

    try:
        while True:
            bs = live["batch"]

            # ── Refresh model weights from server ─────
            if done - last_weight_refresh >= REFRESH_WEIGHTS_EVERY:
                updated = download_model_weights(model, device)
                if updated:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
                    last_weight_refresh = done
                    print(f"  [weights] refreshed from server at batch {done} — gradients now in sync")

            # ── Get training batch from server ────────
            try:
                resp = requests.get(f"{SERVER_URL}/get_batch",
                    params={"worker_id": worker_id, "size": bs}, timeout=30)
            except Exception as e:
                print(f"  [net] batch fetch failed: {e} — retrying in 5s"); time.sleep(5); continue

            if resp.status_code == 204:
                if not _no_data_warned:
                    print("  [wait] Server has no training data yet.")
                    print("         train.py uploads it at startup — make sure it's running.")
                    _no_data_warned = True
                time.sleep(5); continue
            _no_data_warned = False
            if resp.status_code != 200:
                print(f"  [net] server returned HTTP {resp.status_code} — retrying in 5s")
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
                print(f"  [OOM] batch too large — reduced to {live['batch']}"); continue

            # ── Send gradients to server ───────────────
            grads = {}
            for name, param in model.named_parameters():
                if param.grad is None: continue
                g = param.grad.cpu()
                if g.numel() > 10_000:
                    grads[name] = g.half().tolist()
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
                pass   # gradient loss is recoverable — next batch will submit fresh ones

            done += 1; total_loss += loss.item(); avg = total_loss/done
            ela = time.time()-t0; rate = done/ela if ela>0 else 0
            # Show something meaningful — loss going down = your AI is getting smarter
            trend = "↓" if done > 1 and loss.item() < (total_loss-loss.item())/max(done-1,1) else " "
            print(f"  [{done:4d}] loss:{loss.item():.4f} {trend}  avg:{avg:.4f}  "
                  f"batch:{bs}  {rate:.2f}b/s  up:{fmt(ela)}")

    except KeyboardInterrupt:
        print("\n  Stopping — returning your batch allocation to the pool...")
        try:
            requests.post(f"{SERVER_URL}/leave",
                json={"worker_id": worker_id}, timeout=5)
            print(f"  Done! You contributed {done} batches to training your AI.")
            if done > 0:
                print(f"  Final avg loss: {total_loss/done:.4f}  |  session: {fmt(time.time()-t0)}")
        except Exception: pass

if __name__ == "__main__":
    main()
