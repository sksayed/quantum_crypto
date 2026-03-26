from __future__ import annotations

"""
Checks GPU visibility via common ML libraries (PyTorch / TensorFlow / CuPy).

This is best-effort: if the libraries are not installed, it reports that.
"""

import json
from importlib.util import find_spec


def main() -> None:
    report: dict = {"torch": None, "tensorflow": None, "cupy": None}

    # PyTorch
    if find_spec("torch") is None:
        report["torch"] = {"installed": False}
    else:
        import torch

        report["torch"] = {
            "installed": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
        }
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                report["torch"][f"device_{i}"] = {
                    "name": props.name,
                    "total_memory_gib": float(props.total_memory) / (1024**3),
                }

    # TensorFlow
    if find_spec("tensorflow") is None:
        report["tensorflow"] = {"installed": False}
    else:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        report["tensorflow"] = {
            "installed": True,
            "gpus": [str(g) for g in gpus],
        }

    # CuPy
    if find_spec("cupy") is None:
        report["cupy"] = {"installed": False}
    else:
        import cupy as cp

        try:
            count = cp.cuda.runtime.getDeviceCount()
        except Exception as e:  # pragma: no cover
            count = None
            report["cupy"]["error"] = str(e)

        report["cupy"] = {"installed": True, "device_count": count}
        if count and count > 0:
            for i in range(count):
                with cp.cuda.Device(i):
                    props = cp.cuda.runtime.getDeviceProperties(i)
                    # VRAM often available via memGetInfo; best-effort:
                    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
                    report["cupy"][f"device_{i}"] = {
                        "name": props.get("name", b"").decode(errors="ignore"),
                        "total_memory_gib": float(total_mem) / (1024**3),
                    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

