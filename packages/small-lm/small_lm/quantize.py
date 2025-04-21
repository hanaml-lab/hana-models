"""GGUF export through llama.cpp, and the check of what quantisation cost.

``export`` runs llama.cpp's ``convert_hf_to_gguf.py`` and ``llama-quantize`` (set LLAMA_CPP_DIR).
``block_quantize`` reproduces the arithmetic of llama.cpp's symmetric block formats in numpy, so
the error a bit width introduces on a layer can be measured without building anything.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

TYPES = {"Q4_K_M": 4, "Q5_K_M": 5, "Q8_0": 8, "Q4_0": 4}


def block_quantize(w: np.ndarray, bits: int, block: int = 32) -> np.ndarray:
    """Quantise to signed ``bits``-bit integers with one scale per block of ``block`` weights,
    then dequantise. Returns the dequantised array, same shape as ``w``."""
    flat = w.astype(np.float32).ravel()
    pad = (-len(flat)) % block
    blocks = np.pad(flat, (0, pad)).reshape(-1, block)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(blocks).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(blocks / scale), -qmax - 1, qmax)
    return (q * scale).ravel()[: len(flat)].reshape(w.shape)


def relative_error(w: np.ndarray, bits: int, block: int = 32) -> float:
    dq = block_quantize(w, bits, block)
    return float(np.linalg.norm(w - dq) / (np.linalg.norm(w) or 1.0))


def export(model: str, qtype: str, out: str | Path, llama_cpp: str | None = None) -> Path:
    if qtype not in TYPES:
        raise ValueError(f"quant type is one of {', '.join(TYPES)}")
    root = Path(llama_cpp or os.environ.get("LLAMA_CPP_DIR", "llama.cpp"))
    out = Path(out)
    f16 = out.with_suffix(".f16.gguf")
    subprocess.run([sys.executable, str(root / "convert_hf_to_gguf.py"), _local(model), "--outtype", "f16",
                    "--outfile", str(f16)], check=True)
    subprocess.run([str(root / "build/bin/llama-quantize"), str(f16), str(out), qtype], check=True)
    return out


def _local(model: str) -> str:
    if Path(model).exists():
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model)
