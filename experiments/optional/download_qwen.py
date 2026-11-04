"""Download a Qwen small model for the CPU classification probe.
Tries Qwen3.5 small series first, falls back to Qwen3-0.6B.
"""
import sys

from huggingface_hub import snapshot_download

CANDIDATES = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-0.8B-Instruct",
    "Qwen/Qwen3.5-1.7B",
    "Qwen/Qwen3-0.6B",
]

for name in CANDIDATES:
    try:
        print(f"trying {name} ...", flush=True)
        p = snapshot_download(name, ignore_patterns=["*.pth", "*.onnx", "*.gguf"])
        print(f"OK: {name} -> {p}")
        sys.exit(0)
    except Exception as e:
        print(f"  failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

print("ALL CANDIDATES FAILED")
sys.exit(1)
