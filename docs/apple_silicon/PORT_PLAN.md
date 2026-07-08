# bitsandbytes → Apple Silicon (native Metal) Port Plan

**Status:** plan, ready to execute · **Branch:** `feature/mps-metal-kernels` · **Executor:** Fable
**Author of plan:** Claude (Opus 4.8) · **Date:** 2026-07-08

This is an executable spec. It assumes the reader is comfortable in the bitsandbytes
codebase and has hand-written Metal/MSL compute kernels before. **Fable shipped the
CTranslate2 int8 Metal backend** — that is precisely the skill this port needs; this is
_not_ the finetrainers/cogkit "ride `torch.mps`, write no kernels" shape. Here we are
writing real Metal kernels. Line numbers cite a snapshot and **drift** — re-grep the
symbol before editing.

---

## Scoping decisions (locked with Eric)

| Decision     | Value                                                                       | Consequence                                                                                                                                        |
| ------------ | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Path         | **Native C++/Metal kernels** (`csrc/mps_ops.mm` + `csrc/mps_kernels.metal`) | We fill in real MSL kernels + a real Objective-C++ dispatch/load layer, and wire it into Python. Not "improve the pure-PyTorch fallback backend."  |
| First target | **Correctness sweep** (Phase 1 gate)                                        | Before writing a single kernel, build the CPU-oracle parity harness and audit what the `mps` backend actually does. Kernels come _after_ the net.  |
| Correctness  | **CPU-as-oracle, correctness-first**                                        | No CUDA on a Mac. The `default`-backend pure-PyTorch impls are the ground truth. Every native kernel must match CPU within a documented tolerance. |
| Perf         | **Explicitly a later phase**                                                | Get a native kernel that is _correct_ before it is fast. No SIMD-group micro-opt, no GEMM tuning, until parity holds.                              |

---

## §0 — Required reading & house context (reference these; do not re-derive)

**House Apple-Silicon knowledge (skills) — read the matching file before touching Metal:**

- **`apple-silicon` skill** (`~/.claude/skills/apple-silicon/`, references under
  `~/.claude/references/apple-silicon/`) — built for the CTranslate2 Metal backend and
  **directly on-point here** (unlike in the finetrainers port). Load-bearing files:
  - `op-graduation-playbook.md` — the CT2 procedure for graduating an op onto a Metal
    kernel (targeted routing, the fp16 real-kernel-vs-bypass decision + `metal::synchronize()`
    flush nuance, MSL landmines, parity via the existing suite). **Read FIRST before adding
    any kernel.**
    - `compute-kernels-and-dispatch.md` — the full device→library→pipeline→encoder→commit
      chain and threadgroup/grid sizing (`dispatchThreads` vs `dispatchThreadgroups`). This is
      exactly the layer `csrc/mps_ops.mm` is missing.
  - `storage-and-synchronization.md` — Shared storage + unified memory, `flush()`/`synchronize()`
    mechanics, the global-vs-thread-local command-buffer lesson (stale/garbage GPU reads).
  - `mps-matrix-multiplication.md` — `MPSMatrixMultiplication` GEMM, for the 4-bit gemm/gemv
    dequant-then-matmul path if we route matmul through MPS instead of a hand-rolled kernel.
  - `simd-group-functions.md` + `math-functions-and-numeric-parity.md` — for the block-reduction
    (absmax) kernels and CPU-parity tolerances. **`erf` does not exist in MSL**; every bound
    buffer must exist; row-major vs column-major bites.
- **`ct2-internals` skill** — the **op-parity test methodology** (per-backend tolerances,
  CPU-as-oracle). The parity discipline transfers wholesale.

**Prior Apple-Silicon Metal work to mine (this time for the _kernels/dispatch pattern_, not
just discipline):**

- CTranslate2 Metal backend: `/Users/eeaglstun/Documents/dev/CTranslate2/` (`METAL_BACKEND.md`
  at root) — device/library/pipeline lifecycle, command-buffer + autorelease-pool plumbing,
  Shared-storage `contents` access, MPSMatrixMultiplication routing.
- Fable's int8 Metal variant: `/Users/eeaglstun/Documents/dev/CTranslate2-fable-int8/` — the
  closest prior art to bnb's int8 path; reuse the encode/dispatch skeleton, not the math.

**Numeric reality:** fp16/bf16 on MPS produce silently-wrong numbers, not crashes ("confident
garbage"). This is why the correctness sweep (Phase 1) is a hard gate, not a formality.

---

## §1 — Architecture reality (why the plan is shaped this way)

bitsandbytes routes ops through a torch custom-op registry: `bitsandbytes/_ops.py` defines
`torch.library` ops (`bitsandbytes::quantize_4bit`, `::gemm_4bit`, `::int8_linear_matmul`, …);
each backend registers per-device kernels via `register_kernel("bitsandbytes::<op>", "<device>")`.

**The `mps` path today is two divergent, both-incomplete tracks:**

1. **`bitsandbytes/backends/mps/ops.py`** (277 lines — the _only_ path that actually runs):
   registers `mps` kernels for `quantize_blockwise`, `quantize_4bit`, `dequantize_4bit` (+`.out`),
   `gemv_4bit` (+`.out`), `gemm_4bit`. Each op tries a **HuggingFace Hub kernel**
   (`kernels-community/bitsandbytes-mps`, **macOS 26+ only**) and otherwise falls back to
   **pure-PyTorch** (`torch.compile`'d blockwise/4-bit quant; dequant-then-`F.linear` for matmul).
   On macOS < 26 it is fallbacks all the way down.

2. **`csrc/mps_ops.mm` (62 lines) + `csrc/mps_kernels.metal` (117 lines)** — the native path
   this port targets. **It is doubly-dead:**
   - `mps_ops.mm::quantize_mps` literally `NSLog(@"Not implemented"); return nil;`. It has a
     `get_device()`/`get_library()`/`get_graph()` scaffold and loads `bitsandbytes.metallib`, but
     dispatches nothing.
   - `mps_kernels.metal` has exactly one real kernel (`quantize`, a scalar binary-search into a
     256-entry code) and nothing else.
   - **`metallib` appears _nowhere_ in `bitsandbytes/`** — the Python package never loads the
     native MPS library. `cextension.py` only knows how to load CUDA/ROCm/XPU `.so`/`.dylib`s
     (`get_cuda_bnb_library_path`, `CudaBNBNativeLibrary`, `XpuBNBNativeLibrary`); there is **no
     MPS native-library loader and no metallib loader**.

**Build scaffolding _is_ real** (`CMakeLists.txt`): with `-DCOMPUTE_BACKEND=mps` it
`enable_language(OBJCXX)`, compiles `csrc/mps_ops.mm` into `libbitsandbytes_mps.dylib`
(`_mps` suffix, `BUILD_MPS` define), and has a custom command
`xcrun metal -c → build/bitsandbytes.air` then `xcrun metallib → bitsandbytes/bitsandbytes.metallib`,
with `add_dependencies(bitsandbytes metallib)` (~L473) so the metallib builds alongside the lib.

**Therefore the native port = three connected jobs:** (a) real MSL kernels, (b) a real
Objective-C++ dispatch/load layer exposing a stable C ABI, (c) a Python load+call path that
routes the `mps` op registrations to the native lib when present, falling back to today's
Hub/pure-torch path when it isn't. **Phase 1 (correctness sweep) builds the net that makes
(a)–(c) verifiable one kernel at a time.**

---

## §2 — Op surface inventory (re-grep before editing)

Full op surface from `bitsandbytes/_ops.py` (the `torch.library.define` calls) and what each
backend registers:

| Op (`bitsandbytes::…`)                    | CUDA reg? | MPS reg (today, via `ops.py`)           | Native Metal target?                                   |
| ----------------------------------------- | --------- | --------------------------------------- | ------------------------------------------------------ |
| `quantize_blockwise`                      | ✅        | ✅ pure-torch                           | ✅ candidate (the existing `.metal` `quantize` kernel) |
| `dequantize_blockwise` (+`.out`)          | ✅        | ❌ **missing on mps**                   | ✅ candidate                                           |
| `quantize_4bit`                           | ✅        | ✅ Hub / pure-torch                     | ✅ candidate (NF4/FP4 blockwise + pack)                |
| `dequantize_4bit` (+`.out`)               | ✅        | ✅ Hub / pure-torch                     | ✅ candidate                                           |
| `gemv_4bit` (+`.out`)                     | ✅        | ✅ Hub / dequant+linear                 | ⚠️ hard; MPSMatMul or hand kernel — later sub-phase    |
| `gemm_4bit`                               | ✅        | ✅ (M==1 Hub GEMV; else dequant+linear) | ⚠️ hard; later sub-phase                               |
| `int8_linear_matmul` (+`.out`)            | ✅        | ❌ **missing on mps**                   | ⚠️ LLM.int8 — out of Phase-1/2 scope unless Eric adds  |
| `int8_vectorwise_quant`                   | ✅        | ❌ (`default` impl only)                | ⚠️ LLM.int8                                            |
| `int8_vectorwise_dequant`                 | default   | via default                             | —                                                      |
| `int8_mm_dequant`                         | ✅        | ❌                                      | ⚠️ LLM.int8                                            |
| `int8_double_quant`                       | ✅        | ❌                                      | ⚠️ LLM.int8                                            |
| `int8_scaled_mm` / `int8_mixed_scaled_mm` | (compose) | ❌                                      | ⚠️ LLM.int8                                            |
| `optimizer_update_8bit_blockwise`         | ✅        | ❌ **missing on mps**                   | ⚠️ 8-bit optimizers — separate track                   |
| `optimizer_update_32bit`                  | ✅        | ❌                                      | ⚠️ 8-bit optimizers — separate track                   |

**Native-kernel ordering (easiest→hardest, correctness-first):**
`quantize_blockwise` → `dequantize_blockwise` → `dequantize_4bit` → `quantize_4bit` → (later)
`gemv_4bit`/`gemm_4bit`. int8/LLM.int8 and the 8-bit optimizers are **explicitly out of the
first native pass** — flag them, don't build them, unless Eric re-scopes.

---

## §3 — Phased implementation

### Phase 1 — Correctness sweep + parity harness _(the locked first target; gates everything)_

No Metal is written in this phase. Deliver the net.

1. **Audit** — write `docs/apple_silicon/MPS_STATUS.md`: for every `bitsandbytes::` op, record
   (a) does the `mps` backend register it, (b) which path runs on this machine's macOS version
   (Hub vs pure-torch — check `platform.mac_ver()`; Hub is macOS 26+ only), (c) does it match CPU.
2. **Parity harness** — `tests/test_mps_parity.py` (skips cleanly when
   `not torch.backends.mps.is_available()`): for each supported op, generate seeded inputs, run on
   `cpu` (the `default` backend = oracle) and on `mps`, assert allclose within a **documented
   per-dtype tolerance** (fp32 tight; fp16/bf16 looser — follow the `ct2-internals` per-backend
   tolerance convention). Reuse `tests/helpers.py` device/dtype parametrization and mirror the
   existing `test_ops.py` / `test_linear4bit.py` / `test_linear8bitlt.py` structure — do **not**
   invent a new harness shape.
3. **Round-trip checks** — `quantize→dequantize` reconstruction error on MPS vs CPU for NF4/FP4
   and blockwise-int8, across `blocksize ∈ {64,128,256,512}`. This is where "confident garbage"
   first shows.
4. **Baseline record** — capture current pass/fail + tolerances into `MPS_STATUS.md` so each new
   native kernel can be diffed against a known baseline. **Log any op that only "passes" because it
   silently routes to CPU fallback** (`PYTORCH_ENABLE_MPS_FALLBACK`) — that is not real MPS coverage.

**Exit criterion for P1:** `pytest tests/test_mps_parity.py` runs green on an Apple Silicon Mac
(all _currently-supported_ mps ops within tolerance, or documented-xfail with a reason), skips on
non-MPS, and `MPS_STATUS.md` is an accurate ground-truth map. **Nothing native ships until this is
in.**

### Phase 2 — First native kernel end-to-end (prove the whole pipe with ONE op)

Pick **`quantize_blockwise`** (the `.metal` already has a `quantize` kernel to build from). Get
the entire native pipe working for this one op before generalizing:

5. **MSL kernel** — finish/replace `csrc/mps_kernels.metal` blockwise quant: per-block absmax
   reduction + scaled binary-search into the code table, writing packed output + absmax. Follow
   `compute-kernels-and-dispatch.md` for grid/threadgroup sizing and `simd-group-functions.md` for
   the block reduction.
6. **Dispatch layer** — replace the `mps_ops.mm` stub with a real encode path: load the metallib
   (the `get_library()` scaffold is there), build an `MTLComputePipelineState`, get a command
   buffer, bind buffers (input, code, out, absmax, n, blocksize), `dispatchThreads`, commit, and
   synchronize per `storage-and-synchronization.md`. Expose a stable `extern "C"` entry point.
   **Watch the global-vs-thread-local command-buffer footgun and the fp16 flush nuance from the
   op-graduation playbook.**
7. **Python load path** — add MPS native-library loading (extend `cextension.py`: an
   `MpsBNBNativeLibrary` + a loader that finds `libbitsandbytes_mps.dylib` and confirms
   `bitsandbytes.metallib` is present). In `backends/mps/ops.py`, route `quantize_blockwise` to the
   native lib **when it loaded**, else keep today's pure-torch fallback. Never hard-crash when the
   native lib is absent (source installs, wheels without it).
8. **Verify** — the Phase-1 parity test for `quantize_blockwise` now exercises the **native** path
   and stays green. Add a marker/env so the test can assert it hit native (not fallback).

**Exit criterion for P2:** on a source build (`cmake -DCOMPUTE_BACKEND=mps -S . && cmake --build .
&& pip install -e .`), `quantize_blockwise` runs through hand-written Metal, matches CPU within
tolerance, and degrades gracefully to fallback where the native lib is missing.

### Phase 3 — Graduate the remaining quant/dequant ops

9. Repeat the Phase-2 pattern for `dequantize_blockwise`, `dequantize_4bit`, `quantize_4bit`
   (NF4/FP4 pack/unpack, nested absmax where present). Each lands with a green parity test before
   the next starts. `gemv_4bit`/`gemm_4bit` are a **separate later sub-phase** (matmul is the hard
   part — decide MPSMatrixMultiplication-on-dequantized-B vs a fused hand kernel; do not start until
   the quant/dequant ops are solid).

### Phase 4 — Docs + status

10. `docs/apple_silicon/README.md` (user-facing): what the native MPS backend supports, the
    source-build recipe, the supported/unsupported op matrix (from `MPS_STATUS.md`), macOS/torch
    version reality, and the fallback behavior.

---

## §4 — Build & wiring specifics (verified during survey; re-check before editing)

- **Configure/build (Apple Silicon, macOS 14+):**
  `cmake -DCOMPUTE_BACKEND=mps -S .` → `cmake --build . --config Release` → `pip install -e .`.
  Produces `libbitsandbytes_mps.dylib` + `bitsandbytes/bitsandbytes.metallib`.
- **metallib path** — `mps_ops.mm::get_library()` loads `bitsandbytes.metallib` by **relative
  path** (`[NSURL fileURLWithPath:@"bitsandbytes.metallib"]`). That is CWD-relative and will fail
  from an installed package — **fix to resolve next to the loaded dylib / package dir** (mirror how
  `cextension.py`/`consts.py::PACKAGE_DIR` locates native libs).
- **CMake** — `add_dependencies(bitsandbytes metallib)` (~L473) already builds the metallib. Verify
  the metallib and dylib land in the wheel/package dir (`MANIFEST.in`, scikit-build-core packaging)
  so an installed build can find them.
- **No CUDA on Mac** — the oracle is CPU. Do not add CUDA-comparison tests to the MPS suite.

---

## §5 — Out of scope (do not do these now)

- ❌ LLM.int8() native path (`int8_*` ops) — flag as missing; separate scope with Eric.
- ❌ 8-bit optimizer native kernels (`optimizer_update_*`) — separate track.
- ❌ `gemv_4bit`/`gemm_4bit` hand-fused Metal matmul — later sub-phase after quant/dequant are solid.
- ❌ Perf/SIMD micro-optimization, GEMM tuning — correctness first.
- ❌ Removing or regressing the Hub-kernel / pure-torch fallback — the native path **augments** it;
  macOS < 26 and no-native-lib installs must keep working.
- ❌ Touching non-MPS backends (cuda/xpu/hpu/cpu), except read-only as the parity oracle.

---

## §6 — Definition of done (Phase 1–3)

1. `pytest tests/test_mps_parity.py` passes on an Apple Silicon Mac, skips cleanly off-MPS, and
   asserts native-vs-fallback where a native kernel exists.
2. `quantize_blockwise`, `dequantize_blockwise`, `dequantize_4bit`, `quantize_4bit` run through
   **hand-written Metal** and match the CPU oracle within documented per-dtype tolerances.
3. The native path **degrades gracefully**: absent `libbitsandbytes_mps.dylib`/metallib → today's
   Hub/pure-torch fallback, no crash.
4. The metallib loads by an install-safe path (not CWD-relative).
5. `docs/apple_silicon/MPS_STATUS.md` + `README.md` document the op matrix and build recipe.
6. `pre-commit run --all-files` passes (all 10 hooks — ruff, ruff-format, typos, clang-format, …),
   per the repo CLAUDE.md. C++/MSL changes must pass clang-format.

---

## §7 — Risks & open questions for Fable to resolve during execution

- **macOS 26 gate** — the Hub-kernel path is macOS-26-only; on this machine confirm which path the
  baseline actually runs (`platform.mac_ver()`), so the parity harness isn't secretly testing a
  fallback and calling it MPS.
- **CWD-relative metallib load** — the current `get_library()` will not survive an installed
  package; the load-path fix (Phase-2 step 7) is a prerequisite for real use, not a nicety.
- **fp16/bf16 tolerances** — set them from a CPU-oracle round-trip empirically; do not guess. Record
  the chosen tolerances and the torch/macOS versions in `MPS_STATUS.md`.
- **Packaging** — confirm the `.dylib` + `.metallib` are actually included in the built package
  (scikit-build-core + `MANIFEST.in`); a kernel nobody can load is worse than a documented fallback.
- **Upstream drift** — `main` is moving (recent "MPS: improved backend" #1983, ROCm SIMT GEMM #1979).
  Re-grep op signatures in `_ops.py` and the `mps` registrations before editing; rebase awareness.
- **Existing `.metal` `quantize` kernel** — validate its binary-search math against the CPU code
  table before trusting it as the Phase-2 starting point; it predates the current op registry.
