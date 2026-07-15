# MPS Backend Status — Phase 1 audit + Phase 2 first native kernel

**Date:** 2026-07-08 · **Branch:** `feature/mps-metal-kernels` (base: `777c145`)
**Machine:** Apple Silicon (arm64), macOS **26.4.1**
**Stack:** Python 3.14.2 · torch **2.12.1** · bitsandbytes 0.50.0.dev0
**Harness:** `tests/test_mps_parity.py` — Phase-1 baseline (no native build): **183 passed, 1 xfailed
(strict), 0 skipped**. Phase-3 source build (`-DCOMPUTE_BACKEND=mps`, `BNB_MPS_REQUIRE_NATIVE=1`):
**293 passed, 1 xfailed**, with `quantize_blockwise`, `dequantize_blockwise`, `dequantize_4bit`, and
`quantize_4bit` all running through hand-written Metal and bit-exact vs the CPU oracle.

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

| Op (`bitsandbytes::…`)            | `mps` reg?                       | Path that runs here                  | Parity vs CPU oracle                               |
| --------------------------------- | -------------------------------- | ------------------------------------ | -------------------------------------------------- |
| `quantize_blockwise`              | ✅ **native Metal** (P2)         | hand-written kernel (fallback avail) | codes **bit-exact**, absmax **bit-exact**          |
| `dequantize_blockwise`            | ✅ **native Metal** (P3)         | hand-written kernel (fallback avail) | **bit-exact** all dtypes/blocksizes                |
| `dequantize_blockwise.out`        | ❌ (cuda/xpu only, no default)   | **`NotImplementedError`**            | — (gap)                                            |
| `quantize_4bit`                   | ✅ **native Metal** (P3)         | hand-written kernel (fallback avail) | packed nibbles **bit-exact**, absmax **bit-exact** |
| `dequantize_4bit` (+`.out`)       | ✅ **native Metal** (P3)         | hand-written kernel (fallback avail) | **bit-exact** all dtypes/blocksizes                |
| `gemv_4bit` (+`.out`)             | ✅ (dequant native + `F.linear`) | native dequant + `F.linear` on MPS   | fp32 ≤ 7.7e-6; fp16/bf16 0.0 at tested sizes       |
| `gemm_4bit`                       | ✅ (Hub M==1 → dequant+linear)   | dequant + `F.linear` on MPS          | fp32 ≤ 3.9e-6; fp16/bf16 0.0 (incl. nested absmax) |
| `int8_linear_matmul` (+`.out`)    | ❌ → `default`                   | fp32 matmul on MPS                   | exact (int32)                                      |
| `int8_vectorwise_quant`           | ❌ → `default`                   | pure-torch on MPS                    | exact (incl. outlier extraction, threshold=6)      |
| `int8_vectorwise_dequant`         | ❌ → `default`                   | pure-torch on MPS                    | exact                                              |
| `int8_mm_dequant`                 | ❌ → `default`                   | pure-torch on MPS                    | exact                                              |
| `int8_scaled_mm`                  | ❌ → `default`                   | composition of the above             | exact                                              |
| `int8_mixed_scaled_mm`            | ❌ → `default`                   | composition of the above             | covered via components                             |
| `int8_double_quant`               | ❌ (cuda only, no default)       | **`NotImplementedError`**            | — (gap; also unavailable on cpu)                   |
| `optimizer_update_32bit`          | ❌ → `default`                   | pure-torch on MPS                    | exact, **except lion + weight_decay (§5)**         |
| `optimizer_update_8bit_blockwise` | ❌ (cpu/cuda/xpu, no default)    | **`NotImplementedError`**            | — (gap: 8-bit optimizers unusable on mps)          |

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
  **Packaging risk RESOLVED (§9):** the wheel now ships both the `.metallib` and `_mps.dylib` (see
  §9).

Phase 3 (below) graduated dequantize_blockwise + the 4-bit ops onto this exact pipe; the 4-bit
matmuls (`gemv_4bit`/`gemm_4bit`) remain the separate, later hard sub-phase.

---

## 8. Phase 3 — remaining quant/dequant ops on native Metal

**Status: complete and green.** On the source build, three more ops run through hand-written Metal
and are **bit-exact** vs the CPU oracle. Full suite on the native build: **293 passed, 1 xfailed**
(with `BNB_MPS_REQUIRE_NATIVE=1`). Order graduated, each with a green parity test before the next:

| Op                     | New registration?                        | Parity vs CPU oracle (native path)                                            |
| ---------------------- | ---------------------------------------- | ----------------------------------------------------------------------------- |
| `dequantize_blockwise` | **yes** (was missing on mps → `default`) | out **bit-exact** (`torch.equal`), fp32/fp16/bf16 × bs {64,128,256,512}       |
| `dequantize_4bit`      | native swap                              | out **bit-exact**, NF4+FP4 × all dtypes/blocksizes incl. odd-numel tail       |
| `quantize_4bit`        | native swap                              | packed nibbles + absmax **bit-exact**, incl. `quant_storage=bf16` reinterpret |

- **Kernels** (`csrc/mps_kernels.metal`): `dequantize_blockwise` (`out[i]=code[A[i]]*absmax[i/bs]`),
  `dequantize_4bit` (high nibble→even index, low→odd; `out[j]=code4[nib]*absmax[j/bs]`), and
  `quantize_4bit` (per-block absmax + searchsorted over the 15 midpoint bounds of the sorted code +
  `order` remap for FP4 + nibble pack). All one-thread-per-block, fp32 internally; the Python wrapper
  casts dequant output to the requested dtype (matching the reference's trailing `.to(dtype)`), which
  is what makes the fp16/bf16 dequant outputs land **bit-exact** rather than merely within tolerance.
- **Tolerances:** integer/packed outputs (codes, packed nibbles, absmax) asserted **bit-exact**
  (`torch.equal` / view-as-uint8 to dodge NaN≠NaN on the bf16 reinterpret); float dequant outputs use
  the Phase-1 per-dtype tolerances (`assert_parity`) but measured bit-exact in practice.
- **Two reference subtleties reproduced (both were latent bugs risks):**
  1. **Tail-block asymmetry.** The reference stores the tail (partial) block's absmax **clamped** to
     1e-38 and scales it by **direct divide** (`A/absmax`), while full blocks store the **raw** max and
     use **reciprocal-multiply** (`A*(1/absmax)`). Under `-fno-fast-math` these differ by up to 1 ulp,
     which can flip a bucket. The Phase-2 `quantize_blockwise` kernel used reciprocal-multiply for all
     blocks and only passed because test tails were non-zero randn — **hardened in Phase 3** to branch
     on `is_tail` for both `quantize_blockwise` and `quantize_4bit`.
  2. **Odd-numel padding nibble.** For odd numel the reference pads `scaled` with `0.0` and
     **quantizes that** (a nonzero NF4/FP4 index), then packs it as the final low nibble. The kernel
     must quantize `0.0` for that slot, not write a literal `0` — caught by the partial-block test (it
     was a real 1-code mismatch until fixed).
- **`dequantize_4bit` also feeds the matmul fallbacks:** routing it native means `gemv_4bit`/`gemm_4bit`
  on mps now dequantize through Metal before the pure-torch `F.linear`. Their parity tests stay green.
  The matmul itself is untouched (still `F.linear`) — the hard fused-matmul sub-phase has NOT started.

### Sub-task 0 — load-time guard for the `data_ptr()`-is-the-`MTLBuffer` contract

The blind `(__bridge id<MTLBuffer>)` cast rides on an **undocumented** torch internal. Hardened with a
cheap **one-time** check at native-library load (`MpsBNBNativeLibrary.verify_buffer_contract()` →
`extern "C" bnb_mps_check_buffer_contract`): it takes a real MPS tensor's `data_ptr()` + its byte size
and confirms the pointer resolves to a genuine `id<MTLBuffer>` (protocol conformance + `[length]` ≥
size), guarded by `@try/@catch`. If a future torch breaks the contract, `get_mps_library()` **disables
the native path and logs a clear, actionable error** (falls back to pure-torch — no crash, no silent
corruption); `BNB_MPS_REQUIRE_NATIVE=1` then turns that into a hard test failure. Verified: real tensor
→ 1, null pointer → 0, oversize length → 0 (`test_buffer_contract_guard`).

**Unchanged debt (not regressed):** the per-call offset-0 copy of inputs. `gemv_4bit`/`gemm_4bit`
fused matmul and the int8/optimizer ops remain out of scope. (The wheel-packaging gap is now closed
— see §9.)

---

## 9. Packaging — native MPS from a `pip install` (not only source builds)

**Status: resolved.** The wheel now ships both native artifacts and native MPS loads from a plain
`pip install`.

**The gap.** `pyproject.toml` `[tool.setuptools] package-data` matched `libbitsandbytes*.*` — that glob
catches every shared library (all prefixed `lib…`, including `libbitsandbytes_mps.dylib`) but **misses
`bitsandbytes.metallib`**, which has no `lib` prefix. A wheel built before this fix carried the dylib
but not the shader archive, so `get_mps_library()` found the dylib, failed the `metallib.exists()`
gate, and silently fell back to pure-torch.

**The fix (packaging only, one line).** Added a `*.metallib` entry to `package-data`:

```toml
package-data = { "*" = ["libbitsandbytes*.*", "*.metallib", "py.typed"] }
```

Verified the `.dylib` is genuinely covered by the existing glob (not assumed) — see the `unzip -l`
evidence below; both land at `bitsandbytes/…`.

**Build flow (matches how bnb ships prebuilt CUDA `.so`s).** `setup.py`'s `ExtBuildPy` runs a CMake
build (default `COMPUTE_BACKEND=cpu`) during `build_py` **unless `BNB_SKIP_CMAKE=1`**. (`wheel.cmake =
false` is a scikit-build-core _native_-backend setting; this repo uses the `scikit_build_core.setuptools`
shim, where the CMake step is driven by `setup.py` + the `BNB_SKIP_CMAKE` env, so `BNB_SKIP_CMAKE=1` is
the actual switch that skips it.) So the flow is:

```bash
cmake -DCOMPUTE_BACKEND=mps -S . -B . && cmake --build . --config Release   # artifacts -> bitsandbytes/
rm -rf build/ dist/                                                          # avoid staging a stale cpu dylib
BNB_SKIP_CMAKE=1 python -m build --wheel                                     # package pre-built artifacts, no re-run
```

Gotcha found: without `BNB_SKIP_CMAKE=1`, `python -m build` re-runs CMake as `cpu` and adds a stray
`libbitsandbytes_cpu.dylib`; and a stale `build/lib…/` staging dir from an earlier non-skip build gets
swept into the wheel, so clean `build/` first.

**Inclusion proof — `unzip -l dist/*.whl`:**

```
    18074  bitsandbytes/bitsandbytes.metallib
    75928  bitsandbytes/libbitsandbytes_mps.dylib
```

**Runtime proof — isolated throwaway venv (not the source tree).** Fresh venv, `pip install --no-deps`
the built wheel (torch inherited), run from outside the worktree so `import bitsandbytes` resolves to
the _installed_ package. Confirmed against the installed wheel:
`bitsandbytes.__file__` → venv site-packages; both artifacts present in the installed package;
`get_mps_library()` loads native with `metallib_path` resolved (via `dladdr`) inside the venv;
`verify_buffer_contract()` passes; native `quantize_blockwise` bit-exact vs CPU; and the parity subset
`-k "Native or Blockwise8bit or Test4bitParity"` runs **236 passed** with `BNB_MPS_REQUIRE_NATIVE=1`.

**Still open (not this task):** the wheel is a plain-tagged platform wheel; CI matrix / release
automation to actually publish MPS wheels is a separate concern. The per-call offset-0 input copy and
the fused 4-bit matmuls remain as documented above.

---

## 10. Phase M1 — 4-bit matmul baseline + A/B decision (spike)

**Status: measured. Decision made.** This is the `NEXT_MATMUL_PLAN.md` Phase M1 spike, but done
against the _real_ baseline (today's `dequant → F.linear`) rather than an unbuilt native route — the
numbers decide the design fork on their own, so no throwaway MPSMatMul wiring was needed to choose.

**Method.** `torch.mps.synchronize()`-bracketed timing, warmup + 30–50 iters, native dequant forced
(`BNB_MPS_REQUIRE_NATIVE=1`), nf4/blocksize-64. Per shape we isolate the two costs inside today's
unfused path: the native Metal **dequant of B** (materializes full `B_dq`) and the **`F.linear`** GEMM
on that materialized `B_dq`. Bench scripts: `scratchpad/bench_matmul_baseline.py`,
`bench_gemm_baseline.py` (not committed; reproduce from the numbers here).

**gemv (M=1), fp16/bf16** — dequant is the whole cost:

| N     | K     | total  | dequant | linear | dequant share |
| ----- | ----- | ------ | ------- | ------ | ------------- |
| 4096  | 4096  | 0.94ms | 0.75ms  | 0.07ms | ~80%          |
| 11008 | 4096  | 1.80ms | ~2.0ms  | 0.19ms | ~90%+         |
| 4096  | 11008 | 1.81ms | 1.63ms  | 0.21ms | ~90%          |

**gemm (N=K=4096, fp16), sweeping M** — fixed dequant floor, GEMM overtakes it near M≈512:

| M    | total  | dequant | linear | GEMM share |
| ---- | ------ | ------- | ------ | ---------- |
| 8    | 1.41ms | 0.80ms  | 0.10ms | 7%         |
| 64   | 1.21ms | 0.88ms  | 0.42ms | 35%        |
| 512  | 2.50ms | 0.73ms  | 1.27ms | 51%        |
| 2048 | 5.66ms | 0.76ms  | 4.85ms | 86%        |

**Decision (per-op, as the plan anticipated — now with evidence):**

- **`gemv_4bit` (M=1) → Option B (hand-fused dequant+matmul).** 80–90% dequant-bound; Option A
  (`MPSMatrixMultiplication` on materialized `B_dq`) would only touch the ~10% GEMM slice. Fusion —
  never writing `B_dq` to device memory — is the entire win. This is Phase M2.
- **`gemm_4bit` large M (≥~512) → Option A (`MPSMatrixMultiplication`).** GEMM dominates; do not try to
  out-GEMM Apple's tuned kernel by hand. Accept the fixed ~0.75ms dequant tax. This is Phase M3.
- Small-M `gemm` (≤64) is still dequant-bound and behaves like gemv; a fused path helps there too, but
  M3 defaults to Option A for simplicity and lets the fixed dequant floor stand.

> **Phase M2 result (2026-07-14, short note — full docs in Phase M4):** `gemv_4bit` now runs through
> a fused native Metal kernel (`gemv_4bit_fp32/fp16/bf16` in `csrc/mps_kernels.metal`): one
> SIMD-group per output element, uint4 loads of packed B, in-register nf4/fp4 dequant (weights
> rounded to the activation dtype, matching the oracle's `B_dq.to(dtype)`), fp32 fma accumulation
> with split accumulators, `simd_sum` reduction. B_dq is never materialized. Parity: all
> `test_mps_parity.py -k gemv` green under `BNB_MPS_REQUIRE_NATIVE=1` (native asserted via spy);
> fallback preserved (K % 32 != 0 and native-absent paths tested). Wall-clock vs the dequant+
> `F.linear` baseline (nf4/bs64, 50 iters): **3.4–6.2x** across fp16/bf16/fp32 on the M1 shapes
> (e.g. fp16 4096x4096: 0.31ms vs 1.64ms; 11008x4096: 0.51ms vs 1.76ms). Kernel-only GPU time
> (`BNB_MPS_PROFILE=1`, `GPUStartTime/GPUEndTime`): ~0.11ms for the ~25 MB shapes when the GPU is
> clocked up ≈ **~230 GB/s** effective read (vs the standalone dequant kernel's ~54 GB/s); DVFS
> makes idle-gapped calls read ~75 GB/s, and the per-call cross-queue sync + blocking wait
> (~0.15–0.25ms, the Phase-2 discipline) now dominates wall-clock at these sizes. Bench:
> `benchmarks_wip/bench_gemv_fused.py`.

> **Phase M3 result (2026-07-14, short note — full docs in Phase M4):** `gemm_4bit` (general M)
> now runs native per the M1 decision (Option A): `bnb_mps_gemm_4bit` in `csrc/mps_ops.mm` encodes,
> on **one command buffer / one commit / one blocking wait**, (1) a chunked dequant kernel
> (`dequantize_4bit_chunked_fp32/fp16`, one thread per 32-element uint4 chunk, writing
> `(T)(code[nib]*absmax)` into a growable **private scratch MTLBuffer** — same rounding as the
> oracle's `B_dq.to(dtype)`), (2) a shape-cached `MPSMatrixMultiplication` `A[M,K]·B_dq[N,K]ᵀ`
> (row-major, `transposeRight=YES`), and (3) an optional `out[m,n] += bias[n]` epilogue kernel.
> The single sync is the structural win over dequant+`F.linear`, which pays the cross-queue sync
> twice. **bf16 is excluded by the router:** `MPSMatrixMultiplication` hard-asserts on anything but
> fp32/fp16/int8/int16 (probed on macOS 26.4.1: "Input data type must be one of MPSDataTypeFloat32,
> MPSDataTypeFloat16, MPSDataTypeInt8, or MPSDataTypeInt16"), so bf16 keeps the dequant+`F.linear`
> fallback verbatim (still parity-green). Router guards otherwise mirror the fused gemv: K % 32 == 0,
> power-of-two blocksize ≥ 32, packed-size check; nested absmax is unpacked to plain fp32 absmax
> before routing, unchanged. Parity: `test_mps_parity.py -k "gemm_4bit or gemv"` **69 passed** under
> `BNB_MPS_REQUIRE_NATIVE=1` (native asserted via spy incl. ±bias/±nested-absmax; bf16 and
> K%32≠0 fallbacks asserted; fp32 vs MPSMatMul accumulation stayed within the documented 1e-5 atol
> at the calibrated K ≤ 256 — the one tolerance trip found was in the **pure-torch** fallback
> composition at K=256/M=4, so the graceful-fallback test pins K=64 like the main gemm test).
> Wall-clock vs the dequant+`F.linear` fallback (nf4/bs64, N=K=4096, 30 iters):
> fp16 **2.5x** (M=8: 0.47ms vs 1.20ms), **1.5x** (M=64 and M=512), **1.08x** (M=2048);
> fp32 **1.6x/1.5x** (M=8/64), **1.1x** (M=512), **~1.0x** (M=2048). As predicted, the win is the
> single sync + a much faster chunked dequant at small/medium M, and flat at M=2048 where the GEMM
> itself dominates and MPSMatMul ≈ `F.linear`'s GEMM. Bench: `benchmarks_wip/bench_gemm_baseline.py`.

**Load-bearing caveat for the kernel author.** The existing dequant kernel moves ~40 MB in ~0.75ms ≈
**54 GB/s**, on hardware that sustains ~400 GB/s — it's leaving ~85% of memory bandwidth on the floor.
Both matmul routes inherit this: a fused `gemv` kernel that reads packed B no faster than the current
dequant will reproduce the 54 GB/s and win ~nothing. **The M2 target is bandwidth, not "fusion" per se**
— the fused kernel must read packed B + absmax at close to peak bandwidth (coalesced loads, minimal
recompute) or it doesn't beat the baseline. (Separately, this implies the standalone dequant kernel is
itself under-optimized — a possible bigger, simpler lever for the QLoRA M=1 inference case — but that's
Phase-3 kernel scope, out of this phase's remit.)
