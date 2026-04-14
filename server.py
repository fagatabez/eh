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
#   LIVE CONFIG  (edit via /config without restarting)
# ═══════════════════════════════════════════════════

config = {
    "total_batch":      1000,
    "max_vram_pct":     0.70,
    "min_batch":        1,
    # Conservative: 128 tokens * 4 bytes * ~8 (fwd+bwd) = ~4 KB per sample
    # Increase if workers run OOM, decrease to assign more batches
    "bytes_per_sample": 4096,
}

SECRET_KEY = "Dsadasdsefgtgtlubiemlodydsadasdseflubiemlody1bekekejroliwer2011elo%5dfdsfdsk"

# ═══════════════════════════════════════════════════

workers         = {}
free_batch      = config["total_batch"]
lock            = threading.Lock()
gradient_buffer = defaultdict(list)
gradient_lock   = threading.Lock()
training_log    = []
training_stats  = {
    "epoch": 0, "total_epochs": 0, "loss": 0,
    "lr": 0, "status": "idle", "elapsed": 0, "eta": 0,
}
log_lock = threading.Lock()

# CPU bottleneck tracking
cpu_stats = {
    "bottleneck":        False,
    "bottleneck_reason": "",
    "server_cpu_pct":    0.0,
    "pending_gradients": 0,
    "cpu_contributors":  [],   # worker_ids sharing CPU
}
cpu_lock = threading.Lock()

# ── Batch capacity calculation ────────────────────

def vram_to_cap(vram_gb: float) -> int:
    """
    How many samples fit in vram_gb * max_vram_pct?

        usable_bytes = vram_gb * 1024^3 * max_vram_pct
        cap          = usable_bytes // bytes_per_sample

    Example: 8 GB, 70%, 4096 bytes/sample
        = 8 * 1073741824 * 0.70 / 4096
        = 1,468 samples
    """
    usable = vram_gb * (1024 ** 3) * config["max_vram_pct"]
    cap    = int(usable // config["bytes_per_sample"])
    return max(config["min_batch"], cap)

# ── Helpers ───────────────────────────────────────

def redistribute_free_batch():
    global free_batch
    if free_batch <= 0 or not workers:
        return
    eligible = {wid: w for wid, w in workers.items() if w["batch"] < w["cap"]}
    if not eligible:
        return
    per_worker = free_batch // len(eligible)
    if per_worker == 0:
        return
    for wid, w in eligible.items():
        give                  = min(per_worker, w["cap"] - w["batch"])
        workers[wid]["batch"] += give
        free_batch            -= give
    print(f"Redistributed — pool: {free_batch}")

def steal_batches_for_new_worker(needed):
    global free_batch
    available = list(workers.keys())
    random.shuffle(available)
    collected = 0
    for wid in available:
        if collected >= needed:
            break
        if workers[wid]["batch"] > config["min_batch"]:
            workers[wid]["batch"] -= 1
            collected             += 1
    return collected

def cleanup_dead_workers():
    while True:
        time.sleep(10)
        with lock:
            dead = [wid for wid, w in workers.items()
                    if time.time() - w["last_seen"] > 30]
            for wid in dead:
                freed = workers[wid]["batch"]
                del workers[wid]
                with cpu_lock:
                    if wid in cpu_stats["cpu_contributors"]:
                        cpu_stats["cpu_contributors"].remove(wid)
                global free_batch
                free_batch += freed
                print(f"Worker {wid[:8]} timed out — freed {freed} — pool: {free_batch}")
                redistribute_free_batch()

def monitor_cpu():
    """Check server CPU every 5 s and flag bottleneck if > 85%"""
    try:
        import psutil
        has_psutil = True
    except ImportError:
        has_psutil = False
        print("psutil not installed — CPU monitoring disabled. Run: pip install psutil")

    while True:
        time.sleep(5)
        with gradient_lock:
            pending = len(gradient_buffer.get("losses", []))

        if has_psutil:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=1)
        else:
            cpu_pct = 0.0

        bottleneck = cpu_pct > 85 and pending > 10
        with cpu_lock:
            cpu_stats["server_cpu_pct"]    = round(cpu_pct, 1)
            cpu_stats["pending_gradients"] = pending
            cpu_stats["bottleneck"]        = bottleneck
            cpu_stats["bottleneck_reason"] = (
                f"Server CPU at {cpu_pct:.0f}% with {pending} gradient batches queued. "
                "Enable CPU sharing in admin panel to help."
            ) if bottleneck else ""

# ── Worker routes ─────────────────────────────────

@app.route("/join", methods=["POST"])
def join():
    global free_batch
    data       = request.json
    vram_gb    = float(data.get("vram_gb", 4))
    worker_id  = data.get("worker_id", str(random.randint(10000, 99999)))
    gpu_name   = data.get("gpu_name", "Unknown")
    wtype      = data.get("type", "script")
    share_cpu  = bool(data.get("share_cpu", False))
    cpu_pct    = float(data.get("cpu_share_pct", 0.0))

    cap = vram_to_cap(vram_gb)

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
            "batch":         assigned,
            "cap":           cap,
            "vram":          vram_gb,
            "gpu":           gpu_name,
            "type":          wtype,
            "joined":        time.time(),
            "last_seen":     time.time(),
            "share_cpu":     share_cpu,
            "cpu_share_pct": cpu_pct,
        }

    if share_cpu and cpu_pct > 0:
        with cpu_lock:
            if worker_id not in cpu_stats["cpu_contributors"]:
                cpu_stats["cpu_contributors"].append(worker_id)

    usable_mb = vram_gb * 1024 * config["max_vram_pct"]
    print(
        f"Worker {worker_id[:8]} joined | {wtype} | {vram_gb:.1f} GB "
        f"({usable_mb:.0f} MB usable) | cap={cap} | assigned={assigned} | "
        f"pool={free_batch} | cpu_share={cpu_pct:.0f}%"
    )
    return jsonify({
        "worker_id":        worker_id,
        "batch":            assigned,
        "cap":              cap,
        "status":           "ok",
        "bytes_per_sample": config["bytes_per_sample"],
        "max_vram_pct":     config["max_vram_pct"],
    })

@app.route("/ping", methods=["POST"])
def ping():
    data      = request.json
    worker_id = data.get("worker_id")
    with lock:
        if worker_id in workers:
            workers[worker_id]["last_seen"] = time.time()
            if "share_cpu" in data:
                workers[worker_id]["share_cpu"]     = data["share_cpu"]
                workers[worker_id]["cpu_share_pct"] = float(data.get("cpu_share_pct", 0))
                with cpu_lock:
                    if data["share_cpu"] and worker_id not in cpu_stats["cpu_contributors"]:
                        cpu_stats["cpu_contributors"].append(worker_id)
                    elif not data["share_cpu"] and worker_id in cpu_stats["cpu_contributors"]:
                        cpu_stats["cpu_contributors"].remove(worker_id)
            return jsonify({
                "batch":  workers[worker_id]["batch"],
                "cap":    workers[worker_id]["cap"],
                "status": "ok",
                "config": config,
            })
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
            with cpu_lock:
                if worker_id in cpu_stats["cpu_contributors"]:
                    cpu_stats["cpu_contributors"].remove(worker_id)
            del workers[worker_id]
            redistribute_free_batch()
            print(f"Worker {worker_id[:8]} left — freed {freed} — pool: {free_batch}")
    return jsonify({"status": "ok"})

@app.route("/status", methods=["GET"])
def status():
    with lock:
        with cpu_lock:
            cpu_info = dict(cpu_stats)
        return jsonify({
            "workers":       len(workers),
            "total_batches": sum(w["batch"] for w in workers.values()),
            "free_batch":    free_batch,
            "total_vram_gb": round(sum(w["vram"] for w in workers.values()), 1),
            "pool_size":     config["total_batch"],
            "cpu":           cpu_info,
            "config":        config,
            "workers_list":  [
                {
                    "id":        wid[:8],
                    "batch":     w["batch"],
                    "cap":       w["cap"],
                    "vram":      w["vram"],
                    "gpu":       w["gpu"],
                    "type":      w["type"],
                    "uptime":    int(time.time() - w["joined"]),
                    "share_cpu": w.get("share_cpu", False),
                    "cpu_pct":   w.get("cpu_share_pct", 0),
                }
                for wid, w in workers.items()
            ]
        })

# ── Model + data routes ───────────────────────────

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
    print("Model updated")
    return jsonify({"status": "ok"})

@app.route("/tokenizer", methods=["GET"])
def get_tokenizer():
    if os.path.exists("tokenizer.json"):
        return jsonify(json.load(open("tokenizer.json")))
    return "No tokenizer yet", 404

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

# ── Gradient routes ───────────────────────────────

@app.route("/submit_gradients", methods=["POST"])
def submit_gradients():
    data = request.json
    with gradient_lock:
        gradient_buffer["losses"].append(data["loss"])
        for name, grad in data["grads"].items():
            if name not in gradient_buffer:
                gradient_buffer[name] = []
            gradient_buffer[name].append(grad)
    print(f"Gradients from {data['worker_id'][:8]} | loss: {data['loss']:.4f}")
    return jsonify({"status": "ok"})

@app.route("/get_gradients", methods=["GET"])
def get_gradients():
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    with gradient_lock:
        if not gradient_buffer.get("losses"):
            return "", 204
        result = {k: v for k, v in gradient_buffer.items()}
        gradient_buffer.clear()
    return jsonify(result)

# ── Training feed ─────────────────────────────────

@app.route("/training_update", methods=["POST"])
def training_update():
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    data = request.json
    with log_lock:
        training_stats.update({k: v for k, v in data.items() if k != "message"})
        msg = data.get("message")
        if msg:
            training_log.append({"time": time.strftime("%H:%M:%S"), "msg": msg})
        if len(training_log) > 200:
            training_log.pop(0)
    return jsonify({"status": "ok"})

@app.route("/training_feed", methods=["GET"])
def training_feed():
    with log_lock:
        return jsonify({"stats": training_stats, "log": training_log[-50:]})

# ── Config (admin only, live — no restart needed) ─

@app.route("/config", methods=["GET"])
def get_config():
    return jsonify({**config, "free_batch": free_batch})

@app.route("/config", methods=["POST"])
def update_config():
    global free_batch
    if request.headers.get("X-Secret-Key") != SECRET_KEY:
        return "Unauthorized", 401
    data = request.json
    with lock:
        if "total_batch" in data:
            old                    = config["total_batch"]
            config["total_batch"]  = int(data["total_batch"])
            free_batch             = max(0, free_batch + (config["total_batch"] - old))
            print(f"Batch pool: {old} → {config['total_batch']}")

        if "max_vram_pct" in data:
            config["max_vram_pct"] = float(data["max_vram_pct"])
            # recalculate caps LIVE — workers are NOT disconnected
            for wid, w in workers.items():
                new_cap    = vram_to_cap(w["vram"])
                w["cap"]   = new_cap
                if w["batch"] > new_cap:
                    excess       = w["batch"] - new_cap
                    w["batch"]   = new_cap
                    free_batch  += excess
            print(f"max_vram_pct → {config['max_vram_pct']*100:.0f}% (caps updated live)")

        if "min_batch" in data:
            config["min_batch"] = int(data["min_batch"])

        if "bytes_per_sample" in data:
            config["bytes_per_sample"] = int(data["bytes_per_sample"])
            # recalculate caps with new bytes_per_sample
            for wid, w in workers.items():
                w["cap"] = vram_to_cap(w["vram"])

    return jsonify({"status": "ok", "config": config, "free_batch": free_batch})

# ── CPU status ────────────────────────────────────

@app.route("/cpu_status", methods=["GET"])
def get_cpu_status():
    with cpu_lock:
        return jsonify(cpu_stats)

# ── Start ─────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=cleanup_dead_workers, daemon=True).start()
    threading.Thread(target=monitor_cpu,          daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    print(f"Server :{port} | pool={config['total_batch']} | "
          f"max_vram={config['max_vram_pct']*100:.0f}% | "
          f"bytes_per_sample={config['bytes_per_sample']}")
    app.run(host="0.0.0.0", port=port)
