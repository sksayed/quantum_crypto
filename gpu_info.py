from __future__ import annotations

"""
GPU + RAM detection utility.

Primary method: `nvidia-smi` (works for NVIDIA GPUs).
Also reports system RAM from `/proc/meminfo`.
"""

import json
import shutil
import subprocess
from pathlib import Path


def read_system_ram_gib() -> float:
    """
    Return total system RAM in GiB by reading /proc/meminfo.
    """
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return float("nan")

    total_kb = None
    for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("MemTotal:"):
            # Example: "MemTotal:       251658240 kB"
            parts = line.split()
            if len(parts) >= 2:
                total_kb = int(parts[1])
            break

    if total_kb is None:
        return float("nan")
    return total_kb / (1024**2)


def nvidia_smi_query() -> dict:
    """
    Query NVIDIA GPUs using nvidia-smi.
    """
    if shutil.which("nvidia-smi") is None:
        return {"nvidia_smi_available": False}

    # Per-GPU fields
    query_fields = ["name", "memory.total", "memory.free", "memory.used"]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(query_fields)}",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True).strip()

    gpus = []
    for line in out.splitlines():
        # Example: "NVIDIA GeForce RTX 4080 SUPER, 16376, 12000, 4376"
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(query_fields):
            continue
        gpu = {
            "name": parts[0],
            "memory_total_mib": float(parts[1]),
            "memory_free_mib": float(parts[2]),
            "memory_used_mib": float(parts[3]),
        }
        gpus.append(gpu)

    # Driver + CUDA version (best-effort)
    driver_version = None
    cuda_version = None
    try:
        driver_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0].strip()
    except Exception:
        pass
    try:
        cuda_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().splitlines()[0].strip()
    except Exception:
        pass

    return {
        "nvidia_smi_available": True,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "gpus": gpus,
    }


def main() -> None:
    info = {
        "system_ram_total_gib": read_system_ram_gib(),
        "gpu": nvidia_smi_query(),
    }

    print(json.dumps(info, indent=2))

    # Human-friendly summary (also useful for logs)
    gpu = info.get("gpu", {})
    if gpu.get("nvidia_smi_available") and gpu.get("gpus"):
        print("\nSummary:")
        print(f"- NVIDIA GPUs detected: {len(gpu['gpus'])}")
        for i, g in enumerate(gpu["gpus"], start=1):
            print(
                f"  GPU {i}: {g['name']} | VRAM total: {g['memory_total_mib'] / 1024:.2f} GiB"
            )
    else:
        print("\nSummary:")
        print("- NVIDIA GPU not detected (or nvidia-smi not available).")
        print(f"- System RAM total: {info['system_ram_total_gib']:.2f} GiB")


if __name__ == "__main__":
    main()

