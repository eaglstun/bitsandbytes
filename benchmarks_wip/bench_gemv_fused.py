"""Phase M2: fused native gemv_4bit vs the Phase-M1 baseline (dequant -> F.linear).

Per shape, times:
  - fused    torch.ops.bitsandbytes.gemv_4bit (routes to the fused Metal kernel)
  - baseline native dequant of B + F.linear (what gemv_4bit did before Phase M2)
and reports ms/iter plus the fused kernel's effective read bandwidth
(packed B + absmax + A, i.e. the memory the fused kernel actually touches).
"""

import time

import torch

from bitsandbytes.backends.mps import ops as mps_ops
import bitsandbytes.functional as F

DEV = "mps"
ITERS = 50
WARMUP = 10

assert mps_ops._native_available(), "native MPS library required for this benchmark"
assert hasattr(mps_ops._mps_native._lib, "bnb_mps_gemv_4bit"), "fused gemv kernel missing"


def sync():
    torch.mps.synchronize()


def timed(fn):
    for _ in range(WARMUP):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    sync()
    return (time.perf_counter() - t0) / ITERS * 1e3  # ms/iter


def bench(N, K, dtype, quant_type="nf4", blocksize=64):
    A = torch.randn(1, 1, K, dtype=dtype, device=DEV)
    B = torch.randn(N, K, dtype=dtype, device=DEV)
    B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, quant_type, torch.uint8)
    code = F.get_4bit_type(quant_type, device=DEV, blocksize=blocksize)

    def fused():
        return torch.ops.bitsandbytes.gemv_4bit(A, B_q, B.shape, absmax, code, blocksize)

    def baseline():
        B_dq = torch.ops.bitsandbytes.dequantize_4bit(B_q, absmax, blocksize, quant_type, list(B.shape), dtype)
        return torch.nn.functional.linear(A, B_dq)

    # Sanity: fused output matches the baseline it replaces.
    ref = baseline()
    got = fused()
    max_err = (got.float() - ref.float()).abs().max().item()

    t_fused = timed(fused)
    t_base = timed(baseline)

    # Memory the fused kernel reads: packed B (N*K/2 bytes) + absmax (N*K/blocksize fp32)
    # + A (K fp32); writes out (N fp32).
    bytes_moved = N * K // 2 + (N * K // blocksize) * 4 + K * 4 + N * 4
    gbps = bytes_moved / (t_fused * 1e-3) / 1e9

    print(
        f"  N={N:>6} K={K:>6} {str(dtype).replace('torch.', ''):>8}  "
        f"fused={t_fused:7.3f}ms  baseline={t_base:7.3f}ms  "
        f"speedup={t_base / t_fused:5.1f}x  fused-read={gbps:6.1f} GB/s  max|err|={max_err:.2e}"
    )


if __name__ == "__main__":
    print(f"iters={ITERS} warmup={WARMUP} device={DEV}\n")
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        print(f"=== gemv_4bit (M=1), {str(dtype).replace('torch.', '')} ===")
        for N, K in [(4096, 4096), (11008, 4096), (4096, 11008)]:
            bench(N, K, dtype)
        print()
