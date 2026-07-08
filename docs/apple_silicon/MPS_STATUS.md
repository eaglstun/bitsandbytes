# MPS Backend Status — Phase 1 audit + Phase 2 first native kernel

**Date:** 2026-07-08 · **Branch:** `feature/mps-metal-kernels` (base: `777c145`)
**Machine:** Apple Silicon (arm64), macOS **26.4.1**
**Stack:** Python 3.14.2 · torch **2.12.1** · bitsandbytes 0.50.0.dev0
**Harness:** `tests/test_mps_parity.py` — Phase-1 baseline (no native build): **183 passed, 1 xfailed
(strict), 0 skipped**. Phase-2 source build (`-DCOMPUTE_BACKEND=mps`): **199 passed, 1 xfailed**, with
`quantize_blockwise` running through hand-written Metal and bit-exact vs the CPU oracle.

This is the Phase-1 deliverable (§3 audit) plus the Phase-2 result (first native kernel end to end).
Re-verify against `bitsandbytes/_ops.py` and `bitsandbytes/backends/mps/ops.py` before trusting this
after a rebase. The Phase-2 native path is described in §7 below.

---

## 1. How an op resolves on the `mps` device

Three tiers, checked in order by the torch dispatcher:

1. **`mps` registration** (`bitsandbytes/backends/mps/ops.py`) — each such kernel first
   tries the **HuggingFace Hub kernel** (`kernels-community/bitsandbytes-mps`, gated to
   macOS ≥ 26 _and_ requiring the `kernels` package), else falls back to a pure-PyTorch
   implementation executed on mps tensors.
2. **`default` registration** (`bitsandbytes/backends/default/ops.py`) — pure-PyTorch,
   device-agnostic; runs on mps tensors through PyTorch's aten MPS kernels.
3. **Nothing registered** → `NotImplementedError` at call time.

### What actually runs on THIS machine

- **Hub kernels: NEVER run here.** macOS 26.4.1 passes the version gate, but the
  `kernels` package is **not installed**, so `_get_kernel()` fails its import once and
  latches `_kernel_load_failed = True` for the process. Every "Hub-first" op silently
  uses its pure-torch fallback. To exercise the Hub path: `pip install kernels` and
  re-run the parity harness (the same tests then cover it, blocksize-gated).
- **bitsandbytes-native Metal kernels: do not exist yet** (see §4). Nothing in this
  audit exercises `csrc/mps_ops.mm` / `csrc/mps_kernels.metal`.
- **No silent CPU fallback.** `PYTORCH_ENABLE_MPS_FALLBACK` is unset, so any aten op
  missing on MPS would raise instead of quietly routing to CPU. All "parity green"
  results below therefore represent genuine execution on the MPS device (via aten MPS
  kernels) — but **zero** of them represent bitsandbytes-native Metal coverage. This is
  a fallback-quality baseline, which is exactly what Phases 2–3 replace.
- **CPU oracle = `default` backend.** On this source checkout the native library is the
  error-handler mock (`cextension.lib` is `ErrorHandlerMockBNBNativeLibrary`) and the
  host is aarch64 (no AVX512), so none of the lib-gated `cpu` registrations for the
  quant ops exist; the `cpu` device resolves to the same `default` pure-torch kernels.
  Exception: `optimizer_update_32bit` and `optimizer_update_8bit_blockwise` have
  unconditional `cpu` registrations (`backends/cpu/ops.py`), which is how the lion
  divergence in §5 was caught.
- **torch.compile:** the `_try_torch_compile` wrappers compile successfully; alternating
  cpu/mps calls trips dynamo's recompile limit (8) after which execution transparently
  falls back to eager. No correctness impact observed.

---

## 2. Op-by-op matrix (mps device, this machine)

Parity = max deviation vs the CPU oracle with seeded inputs (see §3 for tolerances).

| Op (`bitsandbytes::…`)            | `mps` reg?                     | Path that runs here                  | Parity vs CPU oracle                               |
| --------------------------------- | ------------------------------ | ------------------------------------ | -------------------------------------------------- |
| `quantize_blockwise`              | ✅ **native Metal** (P2)       | hand-written kernel (fallback avail) | codes **bit-exact**, absmax **bit-exact**          |
| `dequantize_blockwise`            | ❌ → `default`                 | pure-torch on MPS                    | exact (0.0) all dtypes/blocksizes                  |
| `dequantize_blockwise.out`        | ❌ (cuda/xpu only, no default) | **`NotImplementedError`**            | — (gap)                                            |
| `quantize_4bit`                   | ✅ (Hub → fallback)            | pure-torch fallback (Hub inert)      | packed nibbles **bit-exact**, absmax exact         |
| `dequantize_4bit` (+`.out`)       | ✅ (Hub → fallback)            | pure-torch fallback                  | exact (0.0) all dtypes/blocksizes                  |
| `gemv_4bit` (+`.out`)             | ✅ (Hub → dequant+`F.linear`)  | dequant + `F.linear` on MPS          | fp32 ≤ 7.7e-6; fp16/bf16 0.0 at tested sizes       |
| `gemm_4bit`                       | ✅ (Hub M==1 → dequant+linear) | dequant + `F.linear` on MPS          | fp32 ≤ 3.9e-6; fp16/bf16 0.0 (incl. nested absmax) |
| `int8_linear_matmul` (+`.out`)    | ❌ → `default`                 | fp32 matmul on MPS                   | exact (int32)                                      |
| `int8_vectorwise_quant`           | ❌ → `default`                 | pure-torch on MPS                    | exact (incl. outlier extraction, threshold=6)      |
| `int8_vectorwise_dequant`         | ❌ → `default`                 | pure-torch on MPS                    | exact                                              |
| `int8_mm_dequant`                 | ❌ → `default`                 | pure-torch on MPS                    | exact                                              |
| `int8_scaled_mm`                  | ❌ → `default`                 | composition of the above             | exact                                              |
| `int8_mixed_scaled_mm`            | ❌ → `default`                 | composition of the above             | covered via components                             |
| `int8_double_quant`               | ❌ (cuda only, no default)     | **`NotImplementedError`**            | — (gap; also unavailable on cpu)                   |
| `optimizer_update_32bit`          | ❌ → `default`                 | pure-torch on MPS                    | exact, **except lion + weight_decay (§5)**         |
| `optimizer_update_8bit_blockwise` | ❌ (cpu/cuda/xpu, no default)  | **`NotImplementedError`**            | — (gap: 8-bit optimizers unusable on mps)          |

**Round-trip reconstruction** (quantize→dequantize on MPS vs same on CPU, seeded randn,
blocksize ∈ {64, 128, 256, 512}, dtypes fp32/fp16/bf16):

- blockwise-int8 (dynamic map): mean abs error ~1e-2 on both devices, **identical** to
  the oracle (codes bit-exact ⇒ reconstruction bit-exact).
- NF4 / FP4: max abs error ~0.55–0.71 and mean ~6e-2 on randn — the expected 4-bit
  quantization error — **identical** on CPU and MPS, including tail (partial-block)
  handling with numel % blocksize ≠ 0.

No "confident garbage" was observed anywhere in the current fallback stack.

---

## 3. Tolerances (documented, empirically calibrated)

Used by `tests/test_mps_parity.py::assert_parity`; per-dtype (rtol, atol), CT2-style
(fp32 tight, halves looser). Measured headroom on this baseline is large — fp32 matmul
divergence is accumulation-order only (≤ ~8e-6 at K ≤ 256); the looser fp16/bf16 bounds
are chosen so the same harness keeps working when native Metal kernels (fast-math,
different accumulation order) replace the fallbacks in Phase 2+.

| dtype | rtol | atol | observed baseline max deviation |
| ----- | ---- | ---- | ------------------------------- |
| fp32  | 1e-6 | 1e-5 | 7.7e-6 (gemv), 3.9e-6 (gemm)    |
| fp16  | 1e-3 | 1e-2 | 0.0                             |
| bf16  | 1e-2 | 4e-2 | 0.0                             |

Additionally:

- **Quantized artifacts (uint8 codes, packed nibbles) must be bit-exact** — a mismatch
  is a wrong bucket, not a rounding difference (`assert_bit_exact`).
- `absmax`/statistics: fp32 tolerance (observed exact — both paths compute absmax in fp32).
- int8/int32 outputs: exact equality.

fp16/bf16 measuring 0.0 today is _not_ an accident to rely on: both matmul fallback
paths dequantize to the activation dtype and run `F.linear`, whose MPS and CPU results
round identically at these small K. Native kernels will not have this property; the
documented tolerances above are the contract.

---

## 4. Native (`csrc`) path — confirmed doubly dead

Verified against the plan's §1 claims, at `777c145`:

- `csrc/mps_ops.mm` (62 lines): `quantize_mps` is `NSLog(@"Not implemented"); return nil;`.
  `get_library()` loads `bitsandbytes.metallib` by **CWD-relative path** (line 33) —
  will not survive an installed package; must be resolved relative to the dylib/package
  dir in Phase 2.
- `csrc/mps_kernels.metal` (117 lines): exactly one kernel (`quantize`, scalar binary
  search into a 256-entry code table). Its math predates the current op registry and is
  **unvalidated** — validate against the CPU code table before using it as the Phase-2
  starting point.
- `metallib` appears **nowhere** in `bitsandbytes/` Python: no loader, no packaging
  reference. `cextension.py` only handles CUDA/ROCm/XPU libraries; on this machine it
  yields the error-handler mock.
- CMake scaffolding (`-DCOMPUTE_BACKEND=mps` → `libbitsandbytes_mps.dylib` +
  `bitsandbytes/bitsandbytes.metallib`) exists but was **not** built or exercised in
  Phase 1 (per plan: Phase 1 audits the existing backend; no native build required).

---

## 5. Findings / divergences

1. **Lion weight-decay semantics differ between backends** (caught by the harness;
   encoded as a strict `xfail`, `test_lion_weight_decay_backend_divergence`):
   - `default` kernel (used on **mps**): **coupled** decay — `g += p * weight_decay`
     (`backends/default/ops.py`, LION included in `optimizer_id in [0, 1, 2, 4]`).
   - `cpu` kernel (`backends/cpu/ops.py::_optimizer_update_32bit_cpu`) and the **CUDA
     kernel** (`csrc/kernels.cu`, `case LION: p_vals[j] *= (1.0f - lr*weight_decay)`):
     **decoupled** decay, matching the Lion paper.
   - The `default` backend is the outlier ⇒ candidate upstream bug affecting every
     device that relies on the default optimizer path (mps included). Out of Phase-1
     scope to fix; flagged for Eric.
2. **8-bit optimizers are unusable on mps** — `optimizer_update_8bit_blockwise` has no
   mps/default registration and raises `NotImplementedError`.
3. **`int8_double_quant` is CUDA-only** — raises on mps _and_ on cpu.
4. **`dequantize_blockwise.out`** raises on mps (only cuda/xpu register the `.out`
   overload; the non-`.out` variant works via `default`).
5. **The Hub-kernel gate is necessary but not sufficient**: macOS 26 alone doesn't
   enable it; the `kernels` package must be installed. A parity report claiming "MPS
   passes" on a macOS-26 machine may still be testing pure-torch fallbacks (as this
   baseline does). §7 risk from the plan: confirmed, resolved by checking
   `bitsandbytes.backends.mps.ops._kernel` at runtime.

---

## 6. Parity harness

`tests/test_mps_parity.py` — mirrors the `tests/test_ops.py` structure and
`tests/helpers.py` parametrization; skips the whole module when
`torch.backends.mps.is_available()` is false.

```bash
pytest tests/test_mps_parity.py -v --tb=short
```

Coverage: quantize/dequantize_blockwise (bit-exactness, parity, round-trip),
quantize/dequantize_4bit (NF4+FP4 × fp32/fp16/bf16 × blocksize {64,128,256,512},
partial-block tail), gemv_4bit, gemm_4bit (± bias, ± nested/compressed absmax),
the int8 op family, optimizer_update_32bit (adam/momentum/rmsprop/lion), and
loud-failure tests pinning the §5 gaps (if a gap op starts working, its test fails,
forcing this document to be updated).

**Baseline record (2026-07-08, torch 2.12.1, macOS 26.4.1): 183 passed, 1 xfailed
(strict; the lion divergence), 0 skipped, ~7 s** (no native build).

`TestNativeMetalPath` gates the Phase-2 native path: it skips when the native library is
absent, or -- with `BNB_MPS_REQUIRE_NATIVE=1` -- fails hard, so a source-build verification
run cannot silently pass on the fallback. It also proves graceful degradation
(`test_graceful_fallback_when_native_absent` forces the native handle off and confirms the
pure-torch path still works).

---

## 7. Phase 2 — first native kernel end to end (`quantize_blockwise`)

**Status: complete and green.** On a source build (`cmake -DCOMPUTE_BACKEND=mps -S . -B . &&
cmake --build . --config Release`), `bitsandbytes::quantize_blockwise` on `mps` runs through a
hand-written Metal kernel and is **bit-exact** vs the CPU oracle (codes AND absmax) across
fp32/fp16/bf16 × blocksize {64,128,256,512}, including partial-block tails. Full suite on the
native build: **199 passed, 1 xfailed**.

### Validation of the pre-existing kernel (plan §7 / step 1)

The old `csrc/mps_kernels.metal::quantize` kernel was validated before trusting it. Its scalar
binary-search core (`quantize_scalar<false>`) is **mathematically correct** — reimplemented in
Python and checked against `torch.bucketize` over the dynamic map: 0/200 000 mismatches. **But the
kernel as a whole was the wrong shape for the op**: no per-block absmax, no scaling by absmax, it
never writes `absmax`, and it used an unrelated `NUM_BLOCK=4096` grid-stride loop instead of the
op's `blocksize`. It predates the current op registry. → **Replaced**, not reused.

### The three connected pieces

1. **MSL kernel** (`csrc/mps_kernels.metal`, `quantize_blockwise`): one thread per block — per-block
   absmax (serial reduction), `scaled = clamp(A * 1/max(absmax,1e-38), -1, 1)`, then a
   searchsorted-left over the 255 midpoint bounds of the 256-entry code table (reproduces
   `torch.bucketize(..., right=False)`). One-thread-per-block is the correctness-first shape; a
   SIMD-group parallel absmax reduction is deferred to a perf phase (per the plan's "correct before
   fast" rule). Compiled with **`-fno-fast-math`** (CMake) so division is correctly rounded and no
   FMA contraction occurs — this is what makes bucket selection identical to the CPU oracle.
2. **Dispatch layer** (`csrc/mps_ops.mm`): replaced the `NSLog("Not implemented")` stub with a real
   encode path — cached device/queue/library/pipeline singletons, `commandBuffer` →
   `computeCommandEncoder` → bind buffers → `dispatchThreads` → `commit` → `waitUntilCompleted`.
   Stable `extern "C" bnb_mps_quantize_blockwise(code, A, out, absmax, n, blocksize)`.
3. **Python load path** (`cextension.py` `MpsBNBNativeLibrary` + `get_mps_library()`;
   `backends/mps/ops.py` routing): native when the lib + metallib are present, else today's
   pure-torch fallback. Never hard-crashes when absent.

### Key implementation decisions / surprises

- **Buffer bridging with zero libtorch linkage.** The CMake `mps` target links only Metal/MPS
  frameworks, not libtorch — so the classic ATen `getMTLBufferStorage` include path isn't available.
  Empirically confirmed on this machine that **a torch MPS tensor's `data_ptr()` IS its
  `id<MTLBuffer>`** (probed: cast to `id<MTLBuffer>`, `[buffer length]` == tensor byte size, class
  `AGXG16XFamilyBuffer`). So the `.mm` casts the ctypes-passed `void*` straight to `id<MTLBuffer>`.
  Ruled out along the way: `data_ptr()` is **not** page-aligned (offsets 4032/6272/…), so
  `newBufferWithBytesNoCopy` fails; and it is **not** a CPU-readable unified pointer (reads returned
  garbage), so a memcpy-in/out bridge is impossible. The object-pointer bridge is the only one that
  works from a ctypes lib.
- **Offset-0 requirement.** `data_ptr()` equals the buffer object only for a `storage_offset == 0`
  tensor; a view's `data_ptr()` is `buffer + offset` and would cast wrong. The Python wrapper forces
  fresh, contiguous, offset-0 fp32 buffers (`_ensure_native_buffer`) before the call. Cost: a copy of
  A per call (acceptable, correctness-first; a later phase can avoid it).
- **Cross-queue synchronization.** The kernel dispatches on its own `MTLCommandQueue`, not torch's
  MPS stream. `torch.mps.synchronize()` is called **before** the dispatch (torch's writes to A/code
  materialized) and the `.mm` blocks on `waitUntilCompleted` **after** commit (outputs complete
  before torch reads). This is the "flush first" lesson from the op-graduation playbook, adapted to a
  separate queue.
- **Install-safe metallib load (plan §4).** The old `get_library()` loaded `bitsandbytes.metallib`
  by CWD-relative path. Now resolved via `dladdr` on a symbol in this dylib → same directory as the
  loaded `.dylib` (both land in `PACKAGE_DIR`), with a `BNB_MPS_METALLIB` env override and a
  CWD-relative last resort.
- **Build layout.** The metallib custom command writes relative paths, so the build must be
  **in-source** (`-B .`, matching the plan's `cmake -S .` recipe) for the metallib and dylib to land
  together in `bitsandbytes/`. An out-of-tree `-B build/` split them. Both files are gitignored.
  **Packaging risk still open (§4):** neither `MANIFEST.in` nor `pyproject.toml` yet force-includes
  the `.metallib`/`_mps.dylib` into a wheel — fine for an editable source build (what Phase 2
  targets), must be addressed before shipping a wheel.

Phases 3+ (dequantize_blockwise, the 4-bit ops, then the 4-bit matmuls) reuse this exact pipe.
