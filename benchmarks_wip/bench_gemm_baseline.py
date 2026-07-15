"""Phase M1 baseline + Phase M3 comparison for gemm_4bit.

M1 question: as M grows, does the GEMM cost overtake dequant? (Answer: crossover near
M~512; below it gemm is dequant-bound.)

M3 question: what does the native path (chunked dequant -> scratch -> MPSMatrixMultiplication
-> bias, ONE command buffer / ONE sync) buy over the dequant + F.linear fallback (native
dequant wait + torch GEMM + second sync)? `gemm_4bit` routes native automatically when built,
so `native` here is just the op; `fallback` reproduces the old tail verbatim.
"""

import time

import torch

import bitsandbytes.backends.mps.ops as mps_ops
import bitsandbytes.functional as F

DEV = "mps"
ITERS = 30
WARMUP = 8


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
    return (time.perf_counter() - t0) / ITERS * 1e3


def bench(M, N, K, dtype, quant_type="nf4", blocksize=64):
    A = torch.randn(1, M, K, dtype=dtype, device=DEV)
    B = torch.randn(N, K, dtype=dtype, device=DEV)
    B_q, qs = F.quantize_4bit(B, blocksize=blocksize, quant_type=quant_type)

    def native():
        # Routes through bnb_mps_gemm_4bit when the native library is built (fp32/fp16).
        return torch.ops.bitsandbytes.gemm_4bit(A, B_q, list(B.shape), qs.absmax, blocksize, quant_type)

    def dequant_only():
        return torch.ops.bitsandbytes.dequantize_4bit(B_q, qs.absmax, blocksize, quant_type, list(B.shape), dtype)

    def fallback():
        # The pre-M3 tail: native dequant (its own sync) + torch F.linear (torch's queue).
        B_dq = mps_ops._dequantize_4bit_impl(B_q, qs.absmax, blocksize, quant_type, list(B.shape), dtype)
        return torch.nn.functional.linear(A, B_dq)

    B_dq = dequant_only()

    def linear_only():
        return torch.nn.functional.linear(A, B_dq)

    t_nat, t_fb, t_deq, t_lin = timed(native), timed(fallback), timed(dequant_only), timed(linear_only)
    print(
        f"  M={M:>4} N={N:>5} K={K:>5} {str(dtype).replace('torch.', ''):>8}  "
        f"native={t_nat:7.3f}ms  fallback={t_fb:7.3f}ms  ({t_fb / t_nat:4.2f}x)  "
        f"[fallback = dequant {t_deq:6.3f} + linear {t_lin:6.3f}]"
    )


if __name__ == "__main__":
    native = "native" if mps_ops._native_available() else "FALLBACK-ONLY (no native build)"
    print(f"iters={ITERS} warmup={WARMUP} device={DEV} lib={native}\n")
    for dtype in (torch.float16, torch.float32):
        print(f"=== gemm_4bit, N=K=4096, {str(dtype).replace('torch.', '')} ===")
        for M in (8, 64, 512, 2048):
            bench(M, 4096, 4096, dtype)
        print()
