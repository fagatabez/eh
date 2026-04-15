# server.py
import os
import json
import gzip
import time
import random
import threading
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# ═══════════════════════════════════════════════════
#   PERSISTENT ADMIN CONFIG
#   All admin changes are written to config.json so
#   they survive server restarts on Railway.
# ═══════════════════════════════════════════════════

CONFIG_FILE = "admin_config.json"

DEFAULTS = {
    "total_batch":      1000,
    "max_vram_pct":     0.70,
    "bytes_per_sample": 32_212_254,   # ~32 MB per sample
    "min_batch":        1,
}

def _load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            saved = json.load(open(CONFIG_FILE))
            merged = dict(DEFAULTS)
            merged.update(saved)
            return merged
        except Exception:
            pass
    return dict(DEFAULTS)

def _save_config():
    try:
        json.dump({
            "total_batch":      TOTAL_BATCH,
            "max_vram_pct":     MAX_VRAM_PCT,
            "bytes_per_sample": BYTES_PER_SAMPLE,
            "min_batch":        MIN_BATCH,
        }, open(CONFIG_FILE, "w"))
    except Exception:
        pass

_cfg = _load_config()
TOTAL_BATCH      = _cfg["total_batch"]
MAX_VRAM_PCT     = _cfg["max_vram_pct"]
BYTES_PER_SAMPLE = _cfg["bytes_per_sample"]
MIN_BATCH        = _cfg["min_batch"]

SECRET_KEY = "Dsadasdsefgtgtlubiemlodydsadasdseflubiemlody1bekekejroliwer2011elo%5dfdsfdsk"

# ═══════════════════════════════════════════════════

workers         = {}
free_batch      = TOTAL_BATCH
lock            = threading.Lock()
gradient_buffer = defaultdict(list)
gradient_lock   = threading.Lock()
log_lock        = threading.Lock()

# ── Training stats + persistent log ──────────────

LOG_FILE = "training_log.json"

def _load_log():
    if os.path.exists(LOG_FILE):
        try:
            return json.load(open(LOG_FILE))
        except Exception:
            pass
    return []

def _save_log(log):
    try:
        json.dump(log[-200:], open(LOG_FILE, "w"))
    except Exception:
        pass

training_log   = _load_log()
training_stats = {
    "epoch": 0, "total_epochs": 0,
    "step": 0, "total_steps": 0,
    "loss": 0, "lr": 0,
    "status": "offline",
    "elapsed": 0, "eta": 0,
    "last_ping": 0,
}

# ── Helpers ───────────────────────────────────────

def is_trainer_online():
    return (time.time() - training_stats.get("last_ping", 0)) < 30

def current_config_snapshot():
    """Return the live config so workers always get the latest values."""
    return {
        "total_batch":      TOTAL_BATCH,
        "max_vram_pct":     MAX_VRAM_PCT,
        "bytes_per_sample": BYTES_PER_SAMPLE,
        "min_batch":        MIN_BATCH,
    }

def rebalance_workers():
    """
    Fairly redistribute all batches proportionally to each worker's cap.
    Called on join, leave, config change, and cleanup.
    """
    global free_batch
    if not workers:
        free_batch = TOTAL_BATCH
        return

    total_available = min(
        sum(w["batch"] for w in workers.values()) + free_batch,
        TOTAL_BATCH
    )
    total_cap = sum(w["cap"] for w in workers.values())
    if total_cap == 0:
        return

    new_batches = {}
    remaining   = total_available

    for wid, w in workers.items():
        share = int(total_available * w["cap"] / total_cap)
        share = max(MIN_BATCH, min(share, w["cap"]))
        new_batches[wid] = share
        remaining -= share

    for wid in sorted(new_batches, key=lambda k: workers[k]["cap"] - new_batches[k], reverse=True):
        if remaining <= 0:
            break
        headroom = workers[wid]["cap"] - new_batches[wid]
        if headroom > 0:
            give = min(headroom, remaining)
            new_batches[wid] += give
            remaining -= give

    for wid, batch in new_batches.items():
        workers[wid]["batch"] = batch
    free_batch = max(0, remaining)

def recalc_caps():
    """Recalculate worker caps after a config change, then rebalance."""
    for w in workers.values():
        usable = w["vram"] * (1024**3) * MAX_VRAM_PCT
        w["cap"] = max(MIN_BATCH, int(usable // BYTES_PER_SAMPLE))
    rebalance_workers()

def cleanup_dead_workers():
    while True:
        time.sleep(10)
        with lock:
            dead = [wid for wid, w in workers.items()
                    if time.time() - w["last_seen"] > 30]
            for wid in dead:
                print(f"Worker {wid[:8]} timed out — removing")
                del workers[wid]
            if dead:
                rebalance_workers()

        with log_lock:
            if not is_trainer_online() and training_stats["status"] == "training":
                training_stats["status"] = "offline"
                entry = {"time": time.strftime("%H:%M:%S"),
                         "msg": "train.py disconnected — waiting for reconnect..."}
                training_log.append(entry)
                _save_log(training_log)

# ═══════════════════════════════════════════════════
#   WORKER ROUTES
# ═══════════════════════════════════════════════════

@app.route("/join", methods=["POST"])
def join():
    data      = request.json
    vram_gb   = float(data.get("vram_gb", 4))
    worker_id = data.get("worker_id", str(random.randint(10000, 99999)))
    gpu_name  = data.get("gpu_name", "Unknown")
    wtype     = data.get("type", "script")

    usable_bytes = vram_gb * (1024 ** 3) * MAX_VRAM_PCT
    cap = max(MIN_BATCH, int(usable_bytes // BYTES_PER_SAMPLE))

    with lock:
        workers[worker_id] = {
            "batch": 0, "cap": cap, "vram": vram_gb,
            "gpu": gpu_name, "type": wtype,
            "joined": time.time(), "last_seen": time.time(),
        }
        rebalance_workers()
        assigned = workers[worker_id]["batch"]

    print(f"Worker {worker_id[:8]} joined | {wtype} | {vram_gb:.1f}GB | "
          f"cap={cap} | assigned={assigned}")
    return jsonify({
        "worker_id":        worker_id,
        "batch":            assigned,
        "cap":              cap,
        "trainer_online":   is_trainer_online(),
        "status":           "ok",
        # Send full admin config so worker applies it immediately
        "config":           current_config_snapshot(),
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
                # Always return the latest admin config so workers self-update
                "config":         current_config_snapshot(),
                "status":         "ok",
            })
    return jsonify({"status": "not_found"}), 404

@app.route("/leave", methods=["POST"])
def leave():
    data      = request.json
    worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            freed = workers[worker_id]["batch"]
            del workers[worker_id]
            rebalance_workers()
            print(f"Worker {worker_id[:8]} left — freed {freed}")
    return jsonify({"status": "ok"})

@app.route("/status", methods=["GET"])
def status():
    with lock:
        return jsonify({
            "workers":        len(workers),
            "total_batches":  sum(w["batch"] for w in workers.values()),
            "free_batch":     free_batch,
            "total_vram_gb":  round(sum(w["vram"] for w in workers.values()), 1),
            "pool_size":      TOTAL_BATCH,
            "trainer_online": is_trainer_online(),
            "config":         current_config_snapshot(),
            "workers_list": [
                {
                    "id":     wid[:8],
                    "batch":  w["batch"],
                    "cap":    w["cap"],
                    "vram":   w["vram"],
                    "gpu":    w["gpu"],
                    "type":   w["type"],
                    "uptime": int(time.time() - w["joined"]),
                }
                for wid, w in workers.items()
            ],
        })

# ═══════════════════════════════════════════════════
#   MODEL + DATA ROUTES
# ═══════════════════════════════════════════════════

@app.route("/model", methods=["GET"])
def get_model():
    if os.path.exists("myai.pt"):
        with open("myai.pt", "rb") as f:
            return f.read(), 200, {"Content-Type": "application/octet-stream"}
    return "No model yet", 404

@app.route("/model", methods=["POST"])
def upload_model():
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    with open("myai.pt", "wb") as f:
        f.write(request.data)
    print(f"Model updated ({len(request.data)/1024/1024:.1f} MB)")
    return jsonify({"status": "ok"})

@app.route("/model", methods=["DELETE"])
def delete_model():
    """
    Called by download_checkpoint.py after it successfully downloads
    myai.pt, so no stale copies pile up on the server.
    """
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    if os.path.exists("myai.pt"):
        os.remove("myai.pt")
        print("Model deleted from server (downloaded by client)")
        return jsonify({"status": "deleted"})
    return jsonify({"status": "not_found"}), 404

@app.route("/tokenizer", methods=["GET"])
def get_tokenizer():
    if os.path.exists("tokenizer.json"):
        with open("tokenizer.json") as f:
            data = json.load(f)
        return jsonify(data)
    # Return proper 404 JSON — fixes the JSONDecodeError in worker.py
    return jsonify({"error": "No tokenizer yet"}), 404

@app.route("/tokenizer", methods=["POST"])
def upload_tokenizer():
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    with open("tokenizer.json", "w") as f:
        json.dump(request.json, f)
    print("Tokenizer updated")
    return jsonify({"status": "ok"})

@app.route("/get_batch", methods=["GET"])
def get_batch():
    if not os.path.exists("training_data.txt.gz"):
        return "", 204
    with gzip.open("training_data.txt.gz", "rt", encoding="utf-8") as f:
        text = f.read()
    size  = int(request.args.get("size", 32))
    start = random.randint(0, max(0, len(text) - size * 200))
    chunk = text[start: start + size * 200]
    tokens = []
    for i in range(0, min(len(chunk), size * 128), 128):
        row = [ord(c) % 30000 for c in chunk[i:i+128]]
        if len(row) == 128:
            tokens.append(row)
    tokens = tokens[:size]
    if not tokens:
        return "", 204
    return jsonify({"batch_id": str(random.randint(0, 999999)), "tokens": tokens})

@app.route("/training_data", methods=["POST"])
def upload_training_data():
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    with open("training_data.txt.gz", "wb") as f:
        f.write(request.data)
    print(f"Training data uploaded: {len(request.data)/1024/1024:.1f} MB")
    return jsonify({"status": "ok"})

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
    print(f"Gradients from {data['worker_id'][:8]} | loss: {data['loss']:.4f}")
    return jsonify({"status": "ok"})

@app.route("/get_gradients", methods=["GET"])
def get_gradients():
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    with log_lock:
        training_stats["last_ping"] = time.time()
        if training_stats["status"] == "offline":
            training_stats["status"] = "training"
            entry = {"time": time.strftime("%H:%M:%S"), "msg": "train.py reconnected!"}
            training_log.append(entry)
            _save_log(training_log)
    with gradient_lock:
        if not gradient_buffer.get("losses"):
            return "", 204
        result = {k: v for k, v in gradient_buffer.items()}
        gradient_buffer.clear()
    return jsonify(result)

# ═══════════════════════════════════════════════════
#   TRAINING FEED
# ═══════════════════════════════════════════════════

@app.route("/training_update", methods=["POST"])
def training_update():
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    data = request.json
    with log_lock:
        training_stats.update({k: v for k, v in data.items() if k != "message"})
        training_stats["last_ping"] = time.time()
        msg = data.get("message")
        if msg:
            training_log.append({"time": time.strftime("%H:%M:%S"), "msg": msg})
            if len(training_log) > 200:
                training_log.pop(0)
            _save_log(training_log)
    return jsonify({"status": "ok"})

@app.route("/training_feed", methods=["GET"])
def training_feed():
    with log_lock:
        stats = dict(training_stats)
        stats["trainer_online"] = is_trainer_online()
        if not is_trainer_online() and stats["status"] == "training":
            stats["status"] = "offline"
        return jsonify({"stats": stats, "log": training_log[-50:]})

# ═══════════════════════════════════════════════════
#   ADMIN CONFIG ROUTES
#   Changes are persisted to admin_config.json and
#   broadcast to all workers via their next /ping.
# ═══════════════════════════════════════════════════

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify(current_config_snapshot())

@app.route("/config", methods=["POST"])
def update_config():
    global TOTAL_BATCH, MAX_VRAM_PCT, BYTES_PER_SAMPLE, free_batch, MIN_BATCH
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    data = request.json
    with lock:
        if "total_batch" in data:
            old = TOTAL_BATCH
            TOTAL_BATCH = int(data["total_batch"])
            free_batch  = max(0, free_batch + (TOTAL_BATCH - old))
            print(f"[admin] total_batch: {old} → {TOTAL_BATCH}")
        if "max_vram_pct" in data:
            MAX_VRAM_PCT = float(data["max_vram_pct"])
            print(f"[admin] max_vram_pct: {MAX_VRAM_PCT}")
        if "bytes_per_sample" in data:
            BYTES_PER_SAMPLE = int(data["bytes_per_sample"])
            print(f"[admin] bytes_per_sample: {BYTES_PER_SAMPLE}")
        if "min_batch" in data:
            MIN_BATCH = max(1, int(data["min_batch"]))
            print(f"[admin] min_batch: {MIN_BATCH}")

        # Recalculate every worker's cap and rebalance immediately
        recalc_caps()

    # Persist so changes survive restarts
    _save_config()

    snap = current_config_snapshot()
    print(f"[admin] Config saved & rebalanced — {snap}")
    return jsonify({"status": "ok", **snap})

# ═══════════════════════════════════════════════════
#   START
# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    threading.Thread(target=cleanup_dead_workers, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    print(f"Server on :{port} | pool:{TOTAL_BATCH} | "
          f"bytes/sample:{BYTES_PER_SAMPLE:,} | max_vram:{MAX_VRAM_PCT*100:.0f}%")
    app.run(host="0.0.0.0", port=port)
