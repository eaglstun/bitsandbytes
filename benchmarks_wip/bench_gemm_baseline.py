"""Phase M1 baseline, gemm side: as M grows, does the GEMM cost overtake dequant?

If dequant stays ~fixed (it's B-only, independent of M) while linear grows with M,
there's a crossover M above which Option A (fast on-device GEMM) starts to matter.
Below it, gemm is just as dequant-bound as gemv and Option B wins there too.
"""

import time

import torch

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

    def full():
        return torch.ops.bitsandbytes.gemm_4bit(A, B_q, list(B.shape), qs.absmax, blocksize, quant_type)

    def dequant_only():
        return torch.ops.bitsandbytes.dequantize_4bit(B_q, qs.absmax, blocksize, quant_type, list(B.shape), dtype)

    B_dq = dequant_only()

    def linear_only():
        return torch.nn.functional.linear(A, B_dq)

    t_full, t_deq, t_lin = timed(full), timed(dequant_only), timed(linear_only)
    print(
        f"  M={M:>4} N={N:>5} K={K:>5} {str(dtype).replace('torch.',''):>8}  "
        f"total={t_full:7.3f}ms  dequant={t_deq:7.3f}ms  linear={t_lin:7.3f}ms  "
        f"(linear is {100*t_lin/t_full:4.1f}% of total)"
    )


if __name__ == "__main__":
    print(f"iters={ITERS} warmup={WARMUP} device={DEV}\n")
    for dtype in (torch.float16,):
        print(f"=== gemm_4bit, N=K=4096, {str(dtype).replace('torch.','')} ===")
        for M in (8, 64, 512, 2048):
            bench(M, 4096, 4096, dtype)
        print()
