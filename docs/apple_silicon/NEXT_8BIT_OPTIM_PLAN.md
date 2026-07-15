# Next phase — native 8-bit blockwise optimizers (`optimizer_update_8bit_blockwise`) on Metal

**Status:** plan, ready to dispatch · **Branch:** `feature/mps-8bit-optim` (off fork `main`)
**Prereqs merged:** Phases 1–4 (native quant/dequant) + the 4-bit matmul sub-phase (M1–M4). **Executor:** Fable.

Executable spec in the shape of `PORT_PLAN.md` / `NEXT_MATMUL_PLAN.md`. Assumes the reader has NOT surveyed
the codebase. Line numbers cite a snapshot and **drift** — re-grep before editing. Read `MPS_STATUS.md`
§7–§9 for how the native pipe / buffer bridge / metallib load / packaging already work — this phase reuses
all of it.

---

## §0 — The gap (why this exists)

`optimizer_update_8bit_blockwise` is registered on **cpu, cuda, xpu, triton** — but **not mps** and not
`default`. On mps it raises `NotImplementedError` (pinned by `tests/test_mps_parity.py::TestKnownGapsOnMps::
test_optimizer_update_8bit_blockwise_missing`). So **8-bit optimizers (`Adam8bit`, `Lion8bit`, …) are
unusable on Apple Silicon**. This phase makes them work — first correctly (pure-torch), then fast (Metal).

**Oracle problem to design around:** on this machine `cextension.lib` is the **mock**
(`ErrorHandlerMockBNBNativeLibrary`, aarch64, no CPU lib), so the lib-gated `cpu` registration is INACTIVE
— there is **no CPU oracle available** for this op here. That is exactly why Phase O1 (a device-agnostic
`default` pure-torch impl) comes first: it is the oracle AND the fallback.

---

## §1 — Op contract (re-grep `bitsandbytes/_ops.py` ~L453)

In-place (returns `()`), mutates `p`, `state1`, `state2?`, `absmax1`, `absmax2?`:

```
optimizer_update_8bit_blockwise(str optimizer_name, Tensor g, Tensor p, Tensor state1, Tensor? state2,
  float beta1, beta2, beta3, alpha, eps, int step, float lr, Tensor qmap1, Tensor? qmap2,
  Tensor absmax1, Tensor? absmax2, float weight_decay, gnorm_scale, bool skip_zeros=False) -> ()
```

- **State layout:** `state1`/`state2` are uint8 codes; `qmap1`/`qmap2` are the 256-entry dynamic quant
  maps; `absmax1`/`absmax2` are per-block fp32 maxima. **blocksize = 256** (fixed for 8-bit blockwise).
- **1-state** optimizers (`state2 is None`): `momentum`, `lars`, `rmsprop`, `adagrad`, `lion`.
  **2-state**: `adam`, `lamb` (state1=m, state2=v), `ademamix` (state1 is a stacked [m1,m2], state2=nu).
- `g`/`p` are fp16/bf16/fp32 (same dtype). `g` scaled by `gnorm_scale`.

**The reference math is `bitsandbytes/backends/cpu/ops.py::_optimizer_update_8bit_blockwise_cpu` (~L469).**
Structure: (1) dequant state via qmap+absmax → fp32; (2) `grad = g*gnorm_scale`; (3) per-optimizer update
of `p` and state (adam/ademamix/momentum/lion/rmsprop/adagrad — copy this math EXACTLY); (4) requantize the
updated state → new codes + new per-block absmax. Note decoupled weight decay for adam/lion, coupled for
momentum/rmsprop/adagrad (matches the Lion fix already landed).

---

## §2 — Phased plan

### Phase O1 — device-agnostic `default` pure-torch impl (the oracle + fallback; NO Metal yet)

1. Add a `@register_kernel("bitsandbytes::optimizer_update_8bit_blockwise", "default")` in
   `bitsandbytes/backends/default/ops.py` that ports the cpu reference math but uses **device-agnostic
   torch ops**: `torch.ops.bitsandbytes.dequantize_blockwise` / `quantize_blockwise` for the state
   re/dequant (these resolve to `default` pure-torch on ANY device, incl. cpu and mps) + the same update
   arithmetic. It must run on **cpu** (→ becomes the oracle) and on **mps** (→ becomes the fallback).
2. This alone closes the gap: 8-bit optimizers now WORK on mps (slow). Update
   `test_optimizer_update_8bit_blockwise_missing` — the op no longer raises; convert it to a parity test.
3. Parity harness: add `TestOptimizer8bitBlockwiseParity` to `tests/test_mps_parity.py` mirroring the
   existing `TestOptimizerParity` (§ run both devices, compare `p`, dequantized `state`, `absmax` within
   the documented 8-bit tolerances — state is quantized so NOT bit-exact; params track the fp32 path).
   Cover adam + lion at minimum; ideally all six optimizers, fp16/bf16/fp32.

### Phase O2 — native Metal kernel (prove the pipe: adam 2-state + lion 1-state)

4. MSL kernel(s) in `csrc/mps_kernels.metal`: one threadgroup per block (blocksize 256). Each: dequantize
   the block's state via qmap+absmax (reuse the Phase-3 `dequantize_blockwise` math), run the optimizer
   update in fp32, compute the new per-block absmax (SIMD reduction — reuse the Phase-2 `quantize_blockwise`
   absmax logic), requantize the new state to codes via qmap (binary-search the 256-entry map, same as the
   quant kernel). Handle `optimizer_id` switch (mirror cpu/triton ids: MOMENTUM 0, RMSPROP 1, ADAGRAD 2,
   ADAM 3, LION 4, ADEMAMIX 5). Built `-fno-fast-math`.
5. `extern "C" bnb_mps_optimizer_update_8bit_blockwise(...)` in `csrc/mps_ops.mm` (copy the encode/commit/
   `waitUntilCompleted` skeleton from `bnb_mps_gemm_4bit`; ONE command buffer). argtypes in
   `cextension.py::MpsBNBNativeLibrary` (hasattr-guarded). Route in a new `mps` registration
   (`backends/mps/ops.py`): native when `_native_available()` AND optimizer supported, else the O1 default.
6. Parity: `TestOptimizer8bitBlockwiseParity` asserts native under `BNB_MPS_REQUIRE_NATIVE=1`; graceful
   fallback preserved. Start with adam + lion; guard-fallback the rest.

### Phase O3 — remaining optimizers native (momentum, rmsprop, adagrad, ademamix)

7. Extend the kernel's `optimizer_id` switch. AdEMAMix's stacked `state1=[m1,m2]` + `absmax1` shape needs
   care (2-state-ish, matches the cpu `ndim==2` branch). Keep every optimizer's fallback until its native
   path is parity-green.

### Phase O4 — docs

8. `MPS_STATUS.md` (new §, op matrix row `optimizer_update_8bit_blockwise` ❌→✅), `docs/apple_silicon/
README.md` (8-bit optimizers ❌→✅), `README.md` accelerator row (8-bit Optimizers ❌→✅ — **only if**
   genuinely working+fast; honest caveats otherwise).

---

## §3 — Reuse (do not reinvent)

- Buffer bridge (`data_ptr()` == `id<MTLBuffer>`, offset-0 contract guard), own queue + sync discipline,
  install-safe metallib load, packaging — all done (`MPS_STATUS.md §7–§9`). New kernels just add functions
  to `csrc/mps_kernels.metal`.
- The Phase-2/3 `quantize_blockwise` (absmax reduction + code bucketing) and `dequantize_blockwise` MSL
  kernels ARE the state re/dequant machinery — lift their math into the fused optimizer kernel.
- Dispatch/registration pattern: copy an existing `extern "C"` entry + pipeline + routing (5 ops already
  exist). The `if _native_available(): ... else <default>` routing mirrors the 4-bit matmul ops.

## §4 — Definition of done

1. `optimizer_update_8bit_blockwise` runs on mps (native Metal when available, else pure-torch default),
   matching the oracle within documented 8-bit tolerances for adam/lion/momentum/rmsprop/adagrad/ademamix,
   fp16/bf16/fp32.
2. `tests/test_mps_parity.py` green; native asserted under `BNB_MPS_REQUIRE_NATIVE=1`; fallback preserved.
   The old `test_optimizer_update_8bit_blockwise_missing` gap test is converted (op no longer raises).
3. A recorded speedup vs the pure-torch default on representative sizes.
4. `pre-commit run --all-files` clean. No PR (fork-internal phases).

## §5 — Out of scope

- ❌ LLM.int8() native — separate track. ❌ 4-bit optimizers — not a thing. ❌ Changing the quant/dequant
  numerics (bit-exact; don't touch).
