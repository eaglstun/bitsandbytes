"""Phase M1 baseline: how slow is today's unfused gemv_4bit (dequant -> F.linear) on MPS?

Times, per shape:
  - total  gemv_4bit (native dequant of B + torch F.linear)
  - dequant-only (native _dequantize_4bit_impl of B)  -> the isolable cost
  - F.linear on an already-materialized B_dq          -> the GEMM cost
so we can see where the wall-clock actually goes and whether fusing dequant into
the matmul (Option B) or just moving the GEMM on-device (Option A) is the win.
"""

import time

import torch

import bitsandbytes.functional as F

DEV = "mps"
ITERS = 50
WARMUP = 10


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

    def full():
        return torch.ops.bitsandbytes.gemv_4bit(A, B_q, B.shape, absmax, code, blocksize)

    def dequant_only():
        return torch.ops.bitsandbytes.dequantize_4bit(B_q, absmax, blocksize, quant_type, list(B.shape), dtype)

    B_dq = dequant_only()

    def linear_only():
        return torch.nn.functional.linear(A, B_dq)

    t_full = timed(full)
    t_deq = timed(dequant_only)
    t_lin = timed(linear_only)
    print(
        f"  N={N:>6} K={K:>6} {str(dtype).replace('torch.', ''):>8}  "
        f"total={t_full:7.3f}ms  dequant={t_deq:7.3f}ms  linear={t_lin:7.3f}ms  "
        f"(dequant is {100 * t_deq / t_full:4.1f}% of total)"
    )


if __name__ == "__main__":
    print(f"iters={ITERS} warmup={WARMUP} device={DEV}\n")
    for dtype in (torch.float16, torch.bfloat16):
        print(f"=== gemv_4bit (M=1), {str(dtype).replace('torch.', '')} ===")
        for N, K in [(4096, 4096), (11008, 4096), (4096, 11008)]:
            bench(N, K, dtype)
        print()
