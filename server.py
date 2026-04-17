# server.py
import os, json, gzip, time, random, threading
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════
#   PERSISTENT ADMIN CONFIG
# ═══════════════════════════════════════════════════

CONFIG_FILE = "admin_config.json"
DEFAULTS = {
    "total_batch":      1000,
    "max_vram_pct":     0.70,
    "bytes_per_sample": 32_212_254,
    "min_batch":        1,
    # CPU sharing — admin can allow workers to donate CPU threads
    "allow_cpu_share":  False,
    "cpu_threads":      2,       # max threads a worker may use for CPU training
}

def _load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            saved = json.load(open(CONFIG_FILE))
            return {**DEFAULTS, **saved}
        except Exception:
            pass
    return dict(DEFAULTS)

def _save_config():
    json.dump({k: globals()[k.upper()] if k.upper() in globals() else globals()[k]
               for k in DEFAULTS}, open(CONFIG_FILE, "w"))

_cfg             = _load_config()
TOTAL_BATCH      = _cfg["total_batch"]
MAX_VRAM_PCT     = _cfg["max_vram_pct"]
BYTES_PER_SAMPLE = _cfg["bytes_per_sample"]
MIN_BATCH        = _cfg["min_batch"]
ALLOW_CPU_SHARE  = _cfg["allow_cpu_share"]
CPU_THREADS      = _cfg["cpu_threads"]

SECRET_KEY = "Dsadasdsefgtgtlubiemlodydsadasdseflubiemlody1bekekejroliwer2011elo%5dfdsfdsk"

workers         = {}   # persistent across restarts via worker_sessions
free_batch      = TOTAL_BATCH
lock            = threading.Lock()
gradient_buffer = defaultdict(list)
gradient_lock   = threading.Lock()
log_lock        = threading.Lock()

# ── Persistent worker sessions ────────────────────
# Workers are kept alive across server restarts by saving their
# session to disk. On reconnect they get their batch back.
SESSIONS_FILE = "worker_sessions.json"

def _load_sessions():
    if os.path.exists(SESSIONS_FILE):
        try: return json.load(open(SESSIONS_FILE))
        except Exception: pass
    return {}

def _save_sessions():
    try:
        data = {wid: {k: v for k, v in w.items() if k != "last_seen"}
                for wid, w in workers.items()}
        json.dump(data, open(SESSIONS_FILE, "w"))
    except Exception: pass

# ── Training stats + log ──────────────────────────
LOG_FILE = "training_log.json"

def _load_log():
    if os.path.exists(LOG_FILE):
        try: return json.load(open(LOG_FILE))
        except Exception: pass
    return []

def _save_log(log):
    try: json.dump(log[-200:], open(LOG_FILE, "w"))
    except Exception: pass

training_log = _load_log()
training_stats = {
    "epoch":0,"total_epochs":0,"step":0,"total_steps":0,
    "loss":0,"lr":0,"status":"offline","elapsed":0,"eta":0,"last_ping":0,
}

def is_trainer_online():
    return (time.time() - training_stats.get("last_ping", 0)) < 30

def config_snapshot():
    return {
        "total_batch":      TOTAL_BATCH,
        "max_vram_pct":     MAX_VRAM_PCT,
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "min_batch":        MIN_BATCH,
        "allow_cpu_share":  ALLOW_CPU_SHARE,
        "cpu_threads":      CPU_THREADS,
    }

def rebalance_workers():
    global free_batch
    if not workers:
        free_batch = TOTAL_BATCH; return
    total_available = min(sum(w["batch"] for w in workers.values()) + free_batch, TOTAL_BATCH)
    total_cap = sum(w["cap"] for w in workers.values())
    if total_cap == 0: return
    new_batches = {}; remaining = total_available
    for wid, w in workers.items():
        share = max(MIN_BATCH, min(int(total_available*w["cap"]/total_cap), w["cap"]))
        new_batches[wid] = share; remaining -= share
    for wid in sorted(new_batches, key=lambda k: workers[k]["cap"]-new_batches[k], reverse=True):
        if remaining <= 0: break
        h = workers[wid]["cap"] - new_batches[wid]
        if h > 0:
            give = min(h, remaining); new_batches[wid] += give; remaining -= give
    for wid, b in new_batches.items(): workers[wid]["batch"] = b
    free_batch = max(0, remaining)

def recalc_caps():
    for w in workers.values():
        # Use the real VRAM the worker reported, capped at what the formula gives.
        # This prevents a browser worker that reported 2 GB getting a huge cap
        # when it actually only has 500 MB of usable WebGL memory.
        usable = w["vram"] * (1024**3) * MAX_VRAM_PCT
        raw_cap = max(MIN_BATCH, int(usable // BYTES_PER_SAMPLE))
        # For web workers, trust the cap they send (set during /join from probe),
        # otherwise use formula. Whichever is lower wins (safety first).
        reported_cap = w.get("reported_cap", raw_cap)
        w["cap"] = max(MIN_BATCH, min(raw_cap, reported_cap))
    rebalance_workers()

def cleanup_dead_workers():
    while True:
        time.sleep(10)
        with lock:
            dead = [wid for wid, w in workers.items()
                    if time.time() - w["last_seen"] > 60]   # 60s grace (was 30)
            for wid in dead:
                print(f"Worker {wid[:8]} timed out")
                del workers[wid]
            if dead:
                rebalance_workers(); _save_sessions()
        with log_lock:
            if not is_trainer_online() and training_stats["status"] == "training":
                training_stats["status"] = "offline"
                training_log.append({"time": time.strftime("%H:%M:%S"),
                                     "msg": "train.py disconnected..."})
                _save_log(training_log)

# ═══════════════════════════════════════════════════
#   WORKER ROUTES
# ═══════════════════════════════════════════════════

@app.route("/join", methods=["POST"])
def join():
    data      = request.json
    vram_gb   = float(data.get("vram_gb", 0))
    worker_id = data.get("worker_id") or str(random.randint(10000,99999))
    gpu_name  = data.get("gpu_name", "Unknown")
    wtype     = data.get("type", "script")

    usable_bytes = vram_gb * (1024**3) * MAX_VRAM_PCT
    cap = max(MIN_BATCH, int(usable_bytes // BYTES_PER_SAMPLE)) if vram_gb > 0 else MIN_BATCH
    # Web workers send their probed cap directly so we don't over-allocate
    reported_cap = int(data.get("probed_cap", cap))

    with lock:
        # Reconnect: restore previous batch allocation if session existed
        prev = _load_sessions().get(worker_id)
        workers[worker_id] = {
            "batch":        prev["batch"] if prev else 0,
            "cap":          min(cap, reported_cap),
            "reported_cap": reported_cap,
            "vram":         vram_gb,
            "gpu":          gpu_name,
            "type":         wtype,
            "joined":       time.time(),
            "last_seen":    time.time(),
        }
        rebalance_workers()
        assigned = workers[worker_id]["batch"]
        _save_sessions()

    print(f"Worker {worker_id[:8]} joined | {wtype} | {vram_gb:.1f}GB | assigned={assigned}")
    return jsonify({
        "worker_id": worker_id, "batch": assigned, "cap": cap,
        "trainer_online": is_trainer_online(), "status": "ok",
        "config": config_snapshot(),
    })

@app.route("/ping", methods=["POST"])
def ping():
    data      = request.json
    worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            workers[worker_id]["last_seen"] = time.time()
            return jsonify({
                "batch":          workers[worker_id]["batch"],
                "trainer_online": is_trainer_online(),
                "config":         config_snapshot(),   # ← always latest admin settings
                "status":         "ok",
            })
    # Worker not found — let it rejoin gracefully
    return jsonify({"status": "not_found", "config": config_snapshot()}), 404

@app.route("/leave", methods=["POST"])
def leave():
    data = request.json; worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            del workers[worker_id]
            rebalance_workers(); _save_sessions()
    return jsonify({"status": "ok"})

@app.route("/status", methods=["GET"])
def status():
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_pct = 0.0

    pending_grads = 0
    with gradient_lock:
        pending_grads = len(gradient_buffer.get("losses", []))

    with lock:
        return jsonify({
            "workers":        len(workers),
            "total_batches":  sum(w["batch"] for w in workers.values()),
            "free_batch":     free_batch,
            "total_vram_gb":  round(sum(w["vram"] for w in workers.values()), 1),
            "pool_size":      TOTAL_BATCH,
            "trainer_online": is_trainer_online(),
            "config":         config_snapshot(),
            "cpu": {
                "server_cpu_pct":  round(cpu_pct, 1),
                "pending_gradients": pending_grads,
                "cpu_contributors": [],
                "bottleneck": cpu_pct > 90,
                "bottleneck_reason": "Server CPU over 90% — gradient processing slowed" if cpu_pct > 90 else "",
            },
            "workers_list": [
                {"id": wid[:8], "batch": w["batch"], "cap": w["cap"],
                 "vram": w["vram"], "gpu": w["gpu"], "type": w["type"],
                 "uptime": int(time.time()-w["joined"])}
                for wid, w in workers.items()
            ],
        })

# ═══════════════════════════════════════════════════
#   MODEL + DATA ROUTES
# ═══════════════════════════════════════════════════

@app.route("/model", methods=["GET"])
def get_model():
    if os.path.exists("myai.pt"):
        with open("myai.pt","rb") as f:
            return f.read(), 200, {"Content-Type":"application/octet-stream"}
    return "No model yet", 404

@app.route("/model", methods=["POST"])
def upload_model():
    if request.headers.get("X-Secret-Key") != SECRET_KEY: return "Unauthorized", 401
    with open("myai.pt","wb") as f: f.write(request.data)
    print(f"Model updated ({len(request.data)/1024/1024:.1f} MB)")
    return jsonify({"status":"ok"})

@app.route("/model", methods=["DELETE"])
def delete_model():
    if request.headers.get("X-Secret-Key") != SECRET_KEY: return "Unauthorized", 401
    if os.path.exists("myai.pt"):
        os.remove("myai.pt"); print("Model deleted (downloaded by client)")
        return jsonify({"status":"deleted"})
    return jsonify({"status":"not_found"}), 404

@app.route("/tokenizer", methods=["GET"])
def get_tokenizer():
    if os.path.exists("tokenizer.json"):
        with open("tokenizer.json") as f: return jsonify(json.load(f))
    return jsonify({"error": "No tokenizer yet"}), 404

@app.route("/tokenizer", methods=["POST"])
def upload_tokenizer():
    if request.headers.get("X-Secret-Key") != SECRET_KEY: return "Unauthorized", 401
    with open("tokenizer.json","w") as f: json.dump(request.json, f)
    print("Tokenizer updated")
    return jsonify({"status":"ok"})

@app.route("/has_training_data", methods=["GET"])
def has_training_data():
    ok = os.path.exists("training_data.txt.gz")
    return jsonify({"ok": ok})

# ── Cached tokenizer word2id for get_batch ────────
_tok_cache = {}

def _get_tok():
    global _tok_cache
    if _tok_cache:
        return _tok_cache
    if os.path.exists("tokenizer.json"):
        try:
            with open("tokenizer.json") as f:
                data = json.load(f)
            _tok_cache = data.get("word2id", {})
        except Exception:
            pass
    return _tok_cache

def _tok_encode(text, word2id, seq_len=128):
    """Tokenize a string using the real word2id vocab."""
    import re as _re
    UNK = word2id.get("<unk>", 1)
    BOS = word2id.get("<bos>", 2)
    EOS = word2id.get("<eos>", 3)
    tokens = _re.findall(r"\w+|[^\w\s]", text.lower())
    ids = [BOS] + [word2id.get(t, UNK) for t in tokens] + [EOS]
    # chunk into seq_len+1 slices
    chunks = []
    for i in range(0, len(ids) - seq_len, seq_len // 2):
        chunk = ids[i:i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append(chunk)
    return chunks

@app.route("/get_batch", methods=["GET"])
def get_batch():
    if not os.path.exists("training_data.txt.gz"):
        return "", 204
    size     = int(request.args.get("size", 32))
    seq_len  = 128
    word2id  = _get_tok()

    # Reload tokenizer if it changed on disk (tokenizer.json updated)
    global _tok_cache
    try:
        mtime = os.path.getmtime("tokenizer.json")
        if not hasattr(get_batch, "_tok_mtime") or get_batch._tok_mtime != mtime:
            get_batch._tok_mtime = mtime
            _tok_cache = {}
            word2id = _get_tok()
    except Exception:
        pass

    try:
        with gzip.open("training_data.txt.gz", "rt", encoding="utf-8") as f:
            text = f.read()
    except Exception:
        return "", 204

    # Pick a random chunk of text large enough to yield `size` sequences
    chunk_chars = size * seq_len * 6   # rough estimate: ~6 chars per token
    start = random.randint(0, max(0, len(text) - chunk_chars))
    chunk = text[start : start + chunk_chars]

    if word2id:
        # Use the real tokenizer
        all_chunks = _tok_encode(chunk, word2id, seq_len)
    else:
        # Fallback if tokenizer not available yet: character ordinals mod vocab
        all_chunks = []
        for i in range(0, min(len(chunk), size * seq_len), seq_len):
            row = [ord(c) % 5000 for c in chunk[i:i + seq_len + 1]]
            if len(row) == seq_len + 1:
                all_chunks.append(row)

    tokens = all_chunks[:size]
    if not tokens:
        return "", 204
    return jsonify({"batch_id": str(random.randint(0, 999999)), "tokens": tokens})

@app.route("/training_data", methods=["POST"])
def upload_training_data():
    if request.headers.get("X-Secret-Key") != SECRET_KEY: return "Unauthorized", 401
    with open("training_data.txt.gz","wb") as f: f.write(request.data)
    print(f"Training data uploaded: {len(request.data)/1024/1024:.1f} MB")
    return jsonify({"status":"ok"})

# ═══════════════════════════════════════════════════
#   GRADIENT ROUTES
# ═══════════════════════════════════════════════════

@app.route("/submit_gradients", methods=["POST"])
def submit_gradients():
    data = request.json
    with gradient_lock:
        gradient_buffer["losses"].append(data["loss"])
        for name, grad in data["grads"].items():
            gradient_buffer[name].append(grad)
    print(f"Gradients from {data['worker_id'][:8]} | loss:{data['loss']:.4f}")
    return jsonify({"status":"ok"})

@app.route("/get_gradients", methods=["GET"])
def get_gradients():
    if request.headers.get("X-Secret-Key") != SECRET_KEY: return "Unauthorized", 401
    with log_lock:
        training_stats["last_ping"] = time.time()
        if training_stats["status"] == "offline":
            training_stats["status"] = "training"
            training_log.append({"time":time.strftime("%H:%M:%S"),"msg":"train.py reconnected!"})
            _save_log(training_log)
    with gradient_lock:
        if not gradient_buffer.get("losses"): return "", 204
        result = {k: v for k, v in gradient_buffer.items()}
        gradient_buffer.clear()
    return jsonify(result)

# ═══════════════════════════════════════════════════
#   TRAINING FEED
# ═══════════════════════════════════════════════════

@app.route("/training_update", methods=["POST"])
def training_update():
    if request.headers.get("X-Secret-Key") != SECRET_KEY: return "Unauthorized", 401
    data = request.json
    with log_lock:
        training_stats.update({k: v for k, v in data.items() if k != "message"})
        training_stats["last_ping"] = time.time()
        msg = data.get("message")
        if msg:
            training_log.append({"time":time.strftime("%H:%M:%S"),"msg":msg})
            if len(training_log) > 200: training_log.pop(0)
            _save_log(training_log)
    return jsonify({"status":"ok"})

@app.route("/training_feed", methods=["GET"])
def training_feed():
    with log_lock:
        stats = dict(training_stats)
        stats["trainer_online"] = is_trainer_online()
        if not is_trainer_online() and stats["status"] == "training":
            stats["status"] = "offline"
        return jsonify({"stats": stats, "log": training_log[-50:]})

# ═══════════════════════════════════════════════════
#   ADMIN CONFIG
#   Changes are persisted + broadcast to workers via /ping
# ═══════════════════════════════════════════════════

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify(config_snapshot())

@app.route("/config", methods=["POST"])
def update_config():
    global TOTAL_BATCH,MAX_VRAM_PCT,BYTES_PER_SAMPLE,free_batch,MIN_BATCH,ALLOW_CPU_SHARE,CPU_THREADS
    if request.headers.get("X-Secret-Key") != SECRET_KEY: return "Unauthorized", 401
    data = request.json
    with lock:
        if "total_batch" in data:
            old=TOTAL_BATCH; TOTAL_BATCH=int(data["total_batch"])
            free_batch=max(0,free_batch+(TOTAL_BATCH-old))
            print(f"[admin] total_batch:{old}→{TOTAL_BATCH}")
        if "max_vram_pct"     in data: MAX_VRAM_PCT=float(data["max_vram_pct"]); print(f"[admin] max_vram_pct:{MAX_VRAM_PCT}")
        if "bytes_per_sample" in data: BYTES_PER_SAMPLE=int(data["bytes_per_sample"]); print(f"[admin] bytes_per_sample:{BYTES_PER_SAMPLE}")
        if "min_batch"        in data: MIN_BATCH=max(1,int(data["min_batch"])); print(f"[admin] min_batch:{MIN_BATCH}")
        if "allow_cpu_share"  in data: ALLOW_CPU_SHARE=bool(data["allow_cpu_share"]); print(f"[admin] allow_cpu_share:{ALLOW_CPU_SHARE}")
        if "cpu_threads"      in data: CPU_THREADS=max(1,int(data["cpu_threads"])); print(f"[admin] cpu_threads:{CPU_THREADS}")
        recalc_caps()

    # Persist so changes survive Railway restarts
    try:
        json.dump({"total_batch":TOTAL_BATCH,"max_vram_pct":MAX_VRAM_PCT,
                   "bytes_per_sample":BYTES_PER_SAMPLE,"min_batch":MIN_BATCH,
                   "allow_cpu_share":ALLOW_CPU_SHARE,"cpu_threads":CPU_THREADS},
                  open(CONFIG_FILE,"w"))
    except Exception: pass

    snap = config_snapshot()
    print(f"[admin] Config saved — {snap}")
    return jsonify({"status":"ok", **snap})

# ═══════════════════════════════════════════════════
#   START
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=cleanup_dead_workers, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    print(f"Server :{port} | pool:{TOTAL_BATCH} | cpu_share:{ALLOW_CPU_SHARE} threads:{CPU_THREADS}")
    app.run(host="0.0.0.0", port=port)
