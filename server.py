# server.py
import os
import json
import time
import random
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allows website visitors to connect

# ═══════════════════════════════════════════════════
#   CONFIGURATION
# ═══════════════════════════════════════════════════

TOTAL_BATCH    = 1000   # total batch pool
MAX_VRAM_PCT   = 0.70   # max 70% GPU usage per worker
MIN_BATCH      = 1     # minimum batches per worker
SECRET_KEY     = "myai_secret_123"  # change this!

# ═══════════════════════════════════════════════════

# in-memory state
workers     = {}   # worker_id -> {batch, vram, joined, last_seen}
free_batch  = TOTAL_BATCH
lock        = threading.Lock()

def cleanup_dead_workers():
    """Remove workers that haven't pinged in 30 seconds"""
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
                print(f"Worker {wid[:8]} left — freed {freed} batches — pool: {free_batch}")
                redistribute_free_batch()

def redistribute_free_batch():
    """Give free batches back to existing workers up to their 70% cap"""
    global free_batch
    if free_batch <= 0 or not workers:
        return
    # find workers below their cap
    eligible = {wid: w for wid, w in workers.items()
                if w["batch"] < w["cap"]}
    if not eligible:
        return
    # distribute evenly
    per_worker = free_batch // len(eligible)
    if per_worker == 0:
        return
    for wid, w in eligible.items():
        can_take   = w["cap"] - w["batch"]
        give       = min(per_worker, can_take)
        workers[wid]["batch"] += give
        free_batch             -= give
    print(f"Redistributed — pool now: {free_batch}")

def steal_batches_for_new_worker(needed):
    """Steal 1 batch each from random workers to give to new worker"""
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

@app.route("/join", methods=["POST"])
def join():
    """Worker joins the pool"""
    global free_batch
    data     = request.json
    vram_gb  = float(data.get("vram_gb", 4))
    worker_id = data.get("worker_id", str(random.randint(10000, 99999)))

    # calculate how many batches this GPU can handle at 70%
    SAMPLES_PER_GB = 20
    cap = int(vram_gb * SAMPLES_PER_GB * MAX_VRAM_PCT)
    cap = max(MIN_BATCH, cap)

    with lock:
        if free_batch >= cap:
            # enough free batches
            assigned    = cap
            free_batch -= cap
        elif free_batch > 0:
            # partial from pool
            assigned    = free_batch
            free_batch  = 0
        else:
            # pool empty — steal from others
            assigned = steal_batches_for_new_worker(cap)

        workers[worker_id] = {
            "batch":     assigned,
            "cap":       cap,
            "vram":      vram_gb,
            "joined":    time.time(),
            "last_seen": time.time()
        }

    print(f"Worker {worker_id[:8]} joined — VRAM: {vram_gb:.1f}GB — batch: {assigned} — pool: {free_batch}")
    return jsonify({
        "worker_id": worker_id,
        "batch":     assigned,
        "status":    "ok"
    })

@app.route("/ping", methods=["POST"])
def ping():
    """Worker heartbeat — keeps them alive"""
    data      = request.json
    worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            workers[worker_id]["last_seen"] = time.time()
            return jsonify({"batch": workers[worker_id]["batch"], "status": "ok"})
    return jsonify({"status": "not_found"}), 404

@app.route("/leave", methods=["POST"])
def leave():
    """Worker leaves gracefully"""
    global free_batch
    data      = request.json
    worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            freed       = workers[worker_id]["batch"]
            free_batch += freed
            del workers[worker_id]
            redistribute_free_batch()
            print(f"Worker {worker_id[:8]} left gracefully — freed {freed}")
    return jsonify({"status": "ok"})

@app.route("/status", methods=["GET"])
def status():
    """Public status endpoint — shown on website"""
    with lock:
        total_workers  = len(workers)
        total_batches  = sum(w["batch"] for w in workers.values())
        total_vram     = sum(w["vram"] for w in workers.values())
        return jsonify({
            "workers":       total_workers,
            "total_batches": total_batches,
            "free_batch":    free_batch,
            "total_vram_gb": round(total_vram, 1),
            "pool_size":     TOTAL_BATCH,
            "workers_list":  [
                {
                    "id":    wid[:8],
                    "batch": w["batch"],
                    "vram":  w["vram"],
                    "uptime": int(time.time() - w["joined"])
                }
                for wid, w in workers.items()
            ]
        })

if __name__ == "__main__":
    # start cleanup thread
    t = threading.Thread(target=cleanup_dead_workers, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 5000))
    print(f"Server starting on port {port}")
    print(f"Total batch pool: {TOTAL_BATCH}")
    app.run(host="0.0.0.0", port=port)