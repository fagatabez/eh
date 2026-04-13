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
#   CONFIGURATION
# ═══════════════════════════════════════════════════

TOTAL_BATCH  = 1000
MAX_VRAM_PCT = 0.70
MIN_BATCH    = 10
SECRET_KEY   = "Dsadasdsefgtgtlubiemlodydsadasdseflubiemlody1bekekejroliwer2011elo%5dfdsfdsk"  # change this!

# ═══════════════════════════════════════════════════

workers         = {}
free_batch      = TOTAL_BATCH
lock            = threading.Lock()
gradient_buffer = defaultdict(list)
gradient_lock   = threading.Lock()

# ── Worker pool management ────────────────────────

def cleanup_dead_workers():
    while True:
        time.sleep(10)
        with lock:
            dead = [wid for wid, w in workers.items()
                    if time.time() - w["last_seen"] > 30]
            for wid in dead:
                freed = workers[wid]["batch"]
                del workers[wid]
                global free_batch
                free_batch += freed
                print(f"Worker {wid[:8]} timed out — freed {freed} batches — pool: {free_batch}")
                redistribute_free_batch()

def redistribute_free_batch():
    global free_batch
    if free_batch <= 0 or not workers:
        return
    eligible = {wid: w for wid, w in workers.items()
                if w["batch"] < w["cap"]}
    if not eligible:
        return
    per_worker = free_batch // len(eligible)
    if per_worker == 0:
        return
    for wid, w in eligible.items():
        can_take             = w["cap"] - w["batch"]
        give                 = min(per_worker, can_take)
        workers[wid]["batch"] += give
        free_batch            -= give
    print(f"Redistributed — pool now: {free_batch}")

def steal_batches_for_new_worker(needed):
    global free_batch
    available = list(workers.keys())
    random.shuffle(available)
    collected = 0
    for wid in available:
        if collected >= needed:
            break
        if workers[wid]["batch"] > MIN_BATCH:
            workers[wid]["batch"] -= 1
            collected             += 1
    return collected

# ── Worker routes ─────────────────────────────────

@app.route("/join", methods=["POST"])
def join():
    global free_batch
    data      = request.json
    vram_gb   = float(data.get("vram_gb", 4))
    worker_id = data.get("worker_id", str(random.randint(10000, 99999)))
    gpu_name  = data.get("gpu_name", "Unknown")
    wtype     = data.get("type", "script")

    SAMPLES_PER_GB = 20
    cap = int(vram_gb * SAMPLES_PER_GB * MAX_VRAM_PCT)
    cap = max(MIN_BATCH, cap)

    with lock:
        if free_batch >= cap:
            assigned    = cap
            free_batch -= cap
        elif free_batch > 0:
            assigned    = free_batch
            free_batch  = 0
        else:
            assigned = steal_batches_for_new_worker(cap)

        workers[worker_id] = {
            "batch":     assigned,
            "cap":       cap,
            "vram":      vram_gb,
            "gpu":       gpu_name,
            "type":      wtype,
            "joined":    time.time(),
            "last_seen": time.time()
        }

    print(f"Worker {worker_id[:8]} joined | {wtype} | {vram_gb:.1f}GB | batch: {assigned} | pool: {free_batch}")
    return jsonify({"worker_id": worker_id, "batch": assigned, "status": "ok"})

@app.route("/ping", methods=["POST"])
def ping():
    data      = request.json
    worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            workers[worker_id]["last_seen"] = time.time()
            return jsonify({"batch": workers[worker_id]["batch"], "status": "ok"})
    return jsonify({"status": "not_found"}), 404

@app.route("/leave", methods=["POST"])
def leave():
    global free_batch
    data      = request.json
    worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            freed       = workers[worker_id]["batch"]
            free_batch += freed
            del workers[worker_id]
            redistribute_free_batch()
            print(f"Worker {worker_id[:8]} left — freed {freed} batches — pool: {free_batch}")
    return jsonify({"status": "ok"})

@app.route("/status", methods=["GET"])
def status():
    with lock:
        return jsonify({
            "workers":       len(workers),
            "total_batches": sum(w["batch"] for w in workers.values()),
            "free_batch":    free_batch,
            "total_vram_gb": round(sum(w["vram"] for w in workers.values()), 1),
            "pool_size":     TOTAL_BATCH,
            "workers_list":  [
                {
                    "id":     wid[:8],
                    "batch":  w["batch"],
                    "vram":   w["vram"],
                    "gpu":    w["gpu"],
                    "type":   w["type"],
                    "uptime": int(time.time() - w["joined"])
                }
                for wid, w in workers.items()
            ]
        })

# ── Model + data routes ───────────────────────────

@app.route("/model", methods=["GET"])
def get_model():
    """Send current model weights to worker"""
    if os.path.exists("myai.pt"):
        with open("myai.pt", "rb") as f:
            return f.read(), 200, {"Content-Type": "application/octet-stream"}
    return "No model yet", 404

@app.route("/model", methods=["POST"])
def upload_model():
    """train.py uploads updated model weights"""
    key = request.headers.get("X-Secret-Key")
    if key != SECRET_KEY:
        return "Unauthorized", 401
    with open("myai.pt", "wb") as f:
        f.write(request.data)
    print("Model updated by train.py")
    return jsonify({"status": "ok"})

@app.route("/tokenizer", methods=["GET"])
def get_tokenizer():
    """Send tokenizer to worker"""
    if os.path.exists("tokenizer.json"):
        return jsonify(json.load(open("tokenizer.json")))
    return "No tokenizer yet", 404

@app.route("/tokenizer", methods=["POST"])
def upload_tokenizer():
    """train.py uploads tokenizer"""
    key = request.headers.get("X-Secret-Key")
    if key != SECRET_KEY:
        return "Unauthorized", 401
    with open("tokenizer.json", "w") as f:
        json.dump(request.json, f)
    print("Tokenizer updated by train.py")
    return jsonify({"status": "ok"})

@app.route("/get_batch", methods=["GET"])
def get_batch():
    """Give worker a batch of tokenized training data"""
    if not os.path.exists("training_data.txt.gz"):
        return "", 204
    with gzip.open("training_data.txt.gz", "rt", encoding="utf-8") as f:
        text = f.read()
    size  = int(request.args.get("size", 32))
    start = random.randint(0, max(0, len(text) - size * 200))
    chunk = text[start : start + size * 200]
    tokens = []
    for i in range(0, min(len(chunk), size * 128), 128):
        row = [ord(c) % 30000 for c in chunk[i:i+128]]
        if len(row) == 128:
            tokens.append(row)
    tokens = tokens[:size]
    if not tokens:
        return "", 204
    return jsonify({
        "batch_id": str(random.randint(0, 999999)),
        "tokens":   tokens
    })

@app.route("/training_data", methods=["POST"])
def upload_training_data():
    """train.py uploads training data so workers can get batches"""
    key = request.headers.get("X-Secret-Key")
    if key != SECRET_KEY:
        return "Unauthorized", 401
    with open("training_data.txt.gz", "wb") as f:
        f.write(request.data)
    print(f"Training data uploaded: {len(request.data)/1024/1024:.1f} MB")
    return jsonify({"status": "ok"})

# ── Gradient routes ───────────────────────────────

@app.route("/submit_gradients", methods=["POST"])
def submit_gradients():
    """Receive gradients from worker"""
    data = request.json
    with gradient_lock:
        gradient_buffer["losses"].append(data["loss"])
        for name, grad in data["grads"].items():
            if name not in gradient_buffer:
                gradient_buffer[name] = []
            gradient_buffer[name].append(grad)
    print(f"Gradients from {data['worker_id'][:8]} | loss: {data['loss']:.4f} | params: {len(data['grads'])}")
    return jsonify({"status": "ok"})

@app.route("/get_gradients", methods=["GET"])
def get_gradients():
    """train.py polls this to get accumulated gradients from all workers"""
    key = request.headers.get("X-Secret-Key")
    if key != SECRET_KEY:
        return "Unauthorized", 401
    with gradient_lock:
        if not gradient_buffer.get("losses"):
            return "", 204
        result = {k: v for k, v in gradient_buffer.items()}
        gradient_buffer.clear()
    return jsonify(result)

# ── Start ─────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=cleanup_dead_workers, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    print(f"Server starting on port {port}")
    print(f"Batch pool: {TOTAL_BATCH} | Max VRAM: {MAX_VRAM_PCT*100:.0f}%")
    app.run(host="0.0.0.0", port=port)
