# Next phase — native 4-bit matmul (`gemv_4bit` / `gemm_4bit`) on Metal

**Status:** plan, ready to dispatch cold · **Branch to open:** `feature/mps-matmul` (off origin/main)
**Prereqs merged:** Phases 1–3 (native quantize/dequantize) + packaging. **Executor:** Fable or a fresh session.

This is an executable spec in the shape of `PORT_PLAN.md`. It assumes the reader has NOT surveyed the
codebase yet — everything needed to start is here. Line numbers cite a snapshot and **drift**; re-grep
the symbol before editing. Read `PORT_PLAN.md` for the overall arc and `MPS_STATUS.md` §7–§9 for how the
native pipe, buffer bridge, guard, and packaging already work — this phase reuses all of it.

---

## §0 — Required reading (do not re-derive)

- `MPS_STATUS.md` — ground truth. Especially §7 (the `data_ptr()`-is-the-`MTLBuffer` bridge, cross-queue
  sync, install-safe metallib load), §8 (the kernel/dispatch/routing pattern to copy), Sub-task 0 (the
  load-time buffer-contract guard — already in place, nothing to add).
- **`apple-silicon` skill**, files:
  - `mps-matrix-multiplication.md` — `MPSMatrixMultiplication` GEMM (the "route matmul through MPS"
    option). **Read before choosing the design fork below.**
  - `compute-kernels-and-dispatch.md` — device→library→pipeline→encoder→commit chain and grid/threadgroup
    sizing (for the hand-fused option).
  - `simd-group-functions.md` + `math-functions-and-numeric-parity.md` — SIMD reductions and the
    fast-vs-precise / `-fno-fast-math` parity rules (we already compile the metallib `-fno-fast-math`).
  - `op-graduation-playbook.md` — the flush-first / synchronize discipline.
- `ct2-internals` skill — CPU-as-oracle parity-tolerance methodology (per-dtype).

Prior art to mine for the MPS GEMM route: the CTranslate2 Metal backend
(`/Users/eeaglstun/Documents/dev/CTranslate2/`, `METAL_BACKEND.md`) routes matmul through
`MPSMatrixMultiplication` — reuse its encode/commit skeleton, not its math.

---

## §1 — Current state (what "unfused" means, precisely)

The `mps` backend registers `gemv_4bit`, `gemv_4bit.out`, and `gemm_4bit`
(`bitsandbytes/backends/mps/ops.py`). Today they are **dequantize-then-`F.linear`**:

- `_gemv_4bit_impl` (~L368): tries an inert HF Hub kernel, then falls to
  `B_dq = _dequantize_4bit_impl(...); return torch.nn.functional.linear(A, B_dq)` (~L386–387).
- `gemm_4bit` registration (~L416): unpacks nested/compressed absmax first
  (`dequantize_blockwise` + offset, ~L435–439), then `B_dq = _dequantize_4bit_impl(...);
return F.linear(A, B_dq, bias)` (~L451–452).
- Since Phase 3, `_dequantize_4bit_impl` routes to the **native** Metal dequant kernel when available
  (`MPS_STATUS.md` §8). So the pipeline today is: **native Metal dequant of B → materialize full fp/bf16
  B_dq in memory → PyTorch MPS `F.linear`**.

Op signatures (from `bitsandbytes/_ops.py`, re-grep before editing):

```
gemv_4bit(A, B, int[] shapeB, absmax, code, blocksize) -> Tensor           # (+ .out overload)
gemm_4bit(A, B, int[] shapeB, absmax, blocksize, str quant_type,
          bias?, absmax_8bit?, absmax_code?, absmax_offset?) -> Tensor
```

`A` is fp16/bf16/fp32 activations `[..., K]`; `B` is packed 4-bit weights (uint8 storage, or reinterpreted
bf16/etc.) with logical shape `shapeB = [N, K]`; output is `[..., N]` in `A.dtype`. `gemv_4bit` is the
`M == 1` case; `gemm_4bit` is general `M`. Nested absmax (compressed statistics) appears **only** in
`gemm_4bit` and is already unpacked to a plain per-block fp32 `absmax` before the matmul — the matmul
phase never sees nested absmax.

**Correctness today:** parity tests pass (`tests/test_mps_parity.py::TestMatmul4bitParity`) — fp32
≤ ~8e-6, fp16/bf16 0.0 at tested sizes. So this phase is a **performance** graduation, not a correctness
fix. The bar: stay within the **same documented tolerances** while removing the full-B_dq materialization
and/or the round-trip to PyTorch's GEMM.

---

## §2 — The design fork (decide in Phase M1 with a spike)

Two ways to make the matmul native. Pick per-shape; they are not mutually exclusive.

### Option A — `MPSMatrixMultiplication` on dequantized B (lower risk)

Dequantize B with the existing native kernel into a scratch `MTLBuffer`, then run Apple's tuned
`MPSMatrixMultiplication` (MPS GEMM) `A · B_dqᵀ` on-device, all inside one `.mm` entry point / one command
buffer. Add bias + write output.

- **Pros:** Apple-tuned GEMM (fast, handles fp16/bf16), minimal new MSL, lowest correctness risk, reuses
  `mps-matrix-multiplication.md` directly. Keeps dequant and matmul on **one queue / one commit** →
  removes the current PyTorch hop and its separate sync.
- **Cons:** still materializes a full `B_dq` (no memory win over today); numeric parity depends on
  MPSMatMul's accumulation vs PyTorch's — must be re-validated against the CPU oracle (likely fine within
  fp16/bf16 tol, verify fp32).
- **Best for:** `gemm_4bit` (general M), where a real GEMM dominates.

### Option B — hand-fused dequant + matmul Metal kernel (higher ceiling)

One MSL kernel reads packed 4-bit B + absmax + code and computes the dot products directly, dequantizing
weights **in registers/threadgroup** without ever writing full `B_dq` to device memory.

- **Pros:** no `B_dq` materialization (the real memory/bandwidth win, especially for `gemv_4bit` M==1,
  which is memory-bound); this is what the CUDA `gemv_4bit`/`gemm_4bit` kernels do.
- **Cons:** most work and highest risk — tiling, SIMD-group reductions, fp16/bf16 accumulation, and
  per-block absmax indexing all have to match the oracle within tol; `-fno-fast-math` already set but FMA
  contraction / accumulation order still needs care (`math-functions-and-numeric-parity.md`).
- **Best for:** `gemv_4bit` (M==1) first — it's the simplest fused case (matrix-vector, one output row of
  work per thread/threadgroup) and the biggest bandwidth win.

**Recommended split:** Option B for `gemv_4bit` (M==1, memory-bound, tractable fused kernel), Option A
(`MPSMatrixMultiplication` on native-dequant B) for `gemm_4bit` general M. Do a small **spike** first
(Phase M1) that benchmarks A vs B on representative shapes before committing; correctness gate is the same
either way.

---

## §3 — Phased implementation

### Phase M1 — spike + decision (no production kernel yet)

1. Micro-bench: for `gemv` (M=1, N,K ∈ {4096,11008}) and `gemm` (M ∈ {8,64,512}), time today's
   dequant+`F.linear` vs (A) native-dequant + `MPSMatrixMultiplication`, on fp16/bf16. Confirm the
   MPS GEMM route is faster and within tolerance; record numbers.
2. Decide the A/B split per op (default: B for gemv, A for gemm). Write the decision into `MPS_STATUS.md`.

### Phase M2 — `gemv_4bit` native (prove the fused pipe with M==1)

3. Implement the chosen route for `gemv_4bit`. If Option B: MSL `gemv_4bit` kernel — one threadgroup per
   output element (or per N-tile), each thread dequantizes its slice of packed B (reuse the nibble/absmax
   math from the Phase-3 `dequantize_4bit` kernel) and accumulates `sum_k A[k] * dequant(B[n,k])` in fp32,
   then casts to `A.dtype`. SIMD-group reduce per `simd-group-functions.md`.
4. New `extern "C" bnb_mps_gemv_4bit(...)` in `csrc/mps_ops.mm` (copy the encode/commit/`waitUntilCompleted`
   skeleton from `bnb_mps_quantize_4bit`); new pipeline in the cache; new argtypes in
   `cextension.py::MpsBNBNativeLibrary`. Route in `_gemv_4bit_impl` **when `_native_available()`** else the
   existing dequant+linear fallback (never regress the fallback).
5. Parity: `TestMatmul4bitParity::test_gemv_4bit` already exists — extend / assert it hits native under
   `BNB_MPS_REQUIRE_NATIVE=1`. Bias handling: `gemv_4bit` has no bias; `gemm_4bit` does.

### Phase M3 — `gemm_4bit` native (general M)

6. Implement the chosen route (default: native dequant into scratch buffer + `MPSMatrixMultiplication`,
   bias epilogue). Handle the already-unpacked plain `absmax` (nested absmax is unpacked before this op —
   do not re-handle it). Route in the `gemm_4bit` registration; keep the pure-torch fallback.
7. Parity: `TestMatmul4bitParity::test_gemm_4bit` (± bias, ± compressed/nested absmax) stays green and hits
   native.

### Phase M4 — address the input-copy debt (see §5) + docs

8. Kill the per-call offset-0 input copy where safe (pass `storage_offset` through the ABI and bind at a
   byte offset). 9. Update `MPS_STATUS.md`, `README.md` accelerator row (QLoRA 4-bit 🐢 → ✅ once fused
   and fast), and `docs/apple_silicon/README.md` (gemv/gemm 〰️ → ✅).

---

## §4 — Reuse (do not reinvent)

- **Buffer bridge:** torch MPS `tensor.data_ptr()` **is** the `id<MTLBuffer>` — cast the ctypes `void*`
  directly (`MPS_STATUS.md` §7). The load-time guard (`bnb_mps_check_buffer_contract` /
  `verify_buffer_contract()`) is already in place; no new guard needed.
- **Sync:** own command queue; `torch.mps.synchronize()` before the call, `waitUntilCompleted` after
  (`MPS_STATUS.md` §7). For the MPSMatMul route, dequant + GEMM go on the **same** command buffer.
- **Metallib load:** `dladdr`-relative, install-safe, already done. New kernels just add functions to
  `csrc/mps_kernels.metal` (built `-fno-fast-math`).
- **Packaging:** already ships the dylib + metallib in the wheel (`MPS_STATUS.md` §9) — no change.
- **Dispatch/registration pattern:** copy an existing `extern "C"` entry + `get_pipeline` + `dispatch_*`
  in `mps_ops.mm`, the argtypes block in `cextension.py`, and the `if _native_available(): ... else
<fallback>` routing in `mps/ops.py`. All three already exist for four ops.

---

## §5 — Open debt this phase should also close

**Per-call offset-0 input copy.** Native ops call `_ensure_native_buffer(...)`, which `.contiguous()` and
`.clone()`s any tensor with `storage_offset != 0`, because the `data_ptr()`-as-`MTLBuffer` cast is only
valid at offset 0. For matmul this copies `A` (and any non-offset-0 operand) every call. Fix: pass each
tensor's `storage_offset() * itemsize` through the C ABI and bind with `[enc setBuffer:buf offset:byteOff
atIndex:i]` instead of forcing offset 0 — but first **verify** that a torch MPS view's `data_ptr()` for a
`storage_offset != 0` tensor is `buffer_object + offset` (a miscast) vs the base buffer object; the
Phase-1 probe suggested the former, so binding the base buffer at a byte offset needs the base pointer,
which torch may not expose. If it can't be done safely, **leave the copy and document why** — do not ship a
miscast. This is a correctness-sensitive optimization; treat it as such.

---

## §6 — Definition of done

1. `gemv_4bit` and `gemm_4bit` run through native Metal (fused kernel and/or `MPSMatrixMultiplication`),
   matching the CPU oracle within the documented per-dtype tolerances (fp32 ~1e-5, fp16 ~1e-2, bf16 ~4e-2).
2. `pytest tests/test_mps_parity.py` green; `TestMatmul4bitParity` asserts native under
   `BNB_MPS_REQUIRE_NATIVE=1`; graceful fallback preserved when the native lib is absent.
3. A recorded speedup vs the dequant+`F.linear` baseline on representative shapes (the reason this phase
   exists).
4. Docs updated: `MPS_STATUS.md` (new §), `docs/apple_silicon/README.md` (gemv/gemm status), and the
   `README.md` accelerator row **only if** the result is genuinely fast (else keep 🐢 — do not overclaim).
5. `pre-commit run --all-files` clean (clang-format on `.mm`/`.metal`).

---

## §7 — Out of scope (still)

- ❌ LLM.int8() native path — separate scope.
- ❌ 8-bit optimizer native kernels — separate track.
- ❌ Any change to the quant/dequant kernels' numerics (they are bit-exact; don't touch).
