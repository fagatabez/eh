# setup.py — run this once to install everything
import subprocess
import sys
import os
import platform

PYTHON_VERSION = "3.11"
PACKAGES = [
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
    "datasets",
    "chromadb",
    "duckduckgo-search",
]

def run(cmd):
    print(f"\n>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def check_python():
    v = sys.version_info
    print(f"Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major == 3 and v.minor == 11:
        print("Python 3.11 detected")
        return True
    else:
        print(f"Python {v.major}.{v.minor} detected — recommended is 3.11")
        return False

def install_python_311():
    system = platform.system()
    print(f"Attempting to install Python 3.11 on {system}...")
    if system == "Windows":
        print("Downloading Python 3.11 installer...")
        run('curl -o python311.exe "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"')
        print("Running installer — follow the prompts!")
        print("IMPORTANT: check 'Add Python to PATH' during install!")
        run("python311.exe")
        os.remove("python311.exe")
    elif system == "Linux":
        run("sudo apt-get install -y python3.11 python3.11-pip")
    elif system == "Darwin":
        run("brew install python@3.11")

def install_packages():
    print("\nInstalling packages...")
    for pkg in PACKAGES:
        cmd = f"{sys.executable} -m pip install {pkg}"
        ok  = run(cmd)
        if not ok:
            print(f"WARNING: failed to install {pkg} — trying without CUDA...")
            # fallback to CPU version for torch
            if "torch" in pkg:
                run(f"{sys.executable} -m pip install torch torchvision torchaudio")

def check_gpu():
    print("\nChecking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"GPU found: {name} ({mem:.1f} GB)")
        else:
            print("No GPU found — will train on CPU (slow)")
    except ImportError:
        print("PyTorch not installed yet")

if __name__ == "__main__":
    print("=" * 50)
    print("  MyAI Setup")
    print("=" * 50)

    python_ok = check_python()
    if not python_ok:
        answer = input(f"\nCurrent Python version is not 3.11. Install Python 3.11? (y/n): ")
        if answer.lower() == "y":
            install_python_311()
            print("\nPlease restart this script after Python 3.11 is installed!")
            sys.exit(0)
        else:
            print("Continuing with current Python version...")

    install_packages()
    check_gpu()

    print("\n" + "=" * 50)
    print("Setup complete! You can now run:")
    print("  python download_data.py")
    print("  python train.py")
    print("  python chat.py")
    print("=" * 50)