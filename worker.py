# worker.py — updated to actually train
import os
import uuid
import time
import torch
import requests
import threading
import torch.nn as nn
from model import MyAI, Config
from tokenizer import Tokenizer

# ═══════════════════════════════════════════════════
SERVER_URL   = "https://eh-production.up.railway.app/"
MAX_VRAM_PCT = 0.70
# ═══════════════════════════════════════════════════

def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1024**3
    return 0

def main():
    vram_gb   = get_vram_gb()
    device    = "cuda" if vram_gb > 0 else "cpu"
    worker_id = str(uuid.uuid4())

    print(f"MyAI Worker | {device} | VRAM: {vram_gb:.1f}GB")
    print(f"Connecting to {SERVER_URL}...")

    # join pool
    resp = requests.post(f"{SERVER_URL}/join", json={
        "worker_id": worker_id,
        "vram_gb":   vram_gb,
    }, timeout=10).json()

    batch_size = resp["batch"]
    print(f"Connected! Batch size: {batch_size}\n")

    # load model + tokenizer from server
    print("Downloading model...")
    model_data = requests.get(f"{SERVER_URL}/model").content
    with open("worker_model.pt", "wb") as f:
        f.write(model_data)

    tok = Tokenizer()
    tok_data = requests.get(f"{SERVER_URL}/tokenizer").json()
    tok.word2id = tok_data["word2id"]
    tok.id2word = {int(v): k for k, v in tok.word2id.items()}
    tok.vocab_size = len(tok.word2id)

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    cfg.embed_dim  = 256
    cfg.num_heads  = 8
    cfg.num_layers = 6

    checkpoint = torch.load("worker_model.pt", map_location=device)
    model = MyAI(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=0)

    print("Starting training loop...\n")

    # heartbeat thread
    def heartbeat():
        while True:
            time.sleep(10)
            try:
                r = requests.post(f"{SERVER_URL}/ping",
                    json={"worker_id": worker_id}, timeout=5).json()
                if r["status"] != "ok":
                    print("Server lost connection")
            except:
                pass
    threading.Thread(target=heartbeat, daemon=True).start()

    try:
        while True:
            # get a batch of training data from server
            resp = requests.get(f"{SERVER_URL}/get_batch",
                params={"worker_id": worker_id, "size": batch_size},
                timeout=30)

            if resp.status_code == 204:
                print("No batches available — waiting...")
                time.sleep(5)
                continue

            batch_data = resp.json()
            tokens = torch.tensor(batch_data["tokens"], dtype=torch.long).to(device)

            # train on it
            x = tokens[:, :-1]
            y = tokens[:, 1:]

            with torch.cuda.amp.autocast() if device == "cuda" else torch.no_grad():
                logits = model(x)
                loss   = loss_fn(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # send gradients back to server
            grads = {
                name: param.grad.cpu().tolist()
                for name, param in model.named_parameters()
                if param.grad is not None
            }

            requests.post(f"{SERVER_URL}/submit_gradients", json={
                "worker_id": worker_id,
                "loss":      loss.item(),
                "grads":     grads,
                "batch_id":  batch_data["batch_id"]
            }, timeout=30)

            print(f"Batch done | loss: {loss.item():.4f}")

    except KeyboardInterrupt:
        print("\nStopping...")
        requests.post(f"{SERVER_URL}/leave", json={"worker_id": worker_id})
        print("Done!")

if __name__ == "__main__":
    main()