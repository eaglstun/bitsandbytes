# bitsandbytes on Apple Silicon (Metal / `mps`)

Preview support for Apple Silicon GPUs via native Metal kernels behind the PyTorch `mps`
backend. This page is the user/developer-facing summary; the executable spec and the
ground-truth parity audit live alongside it (see [Further reading](#further-reading)).

## Status at a glance

- **Native, bit-exact:** 4-bit (NF4/FP4) and 8-bit blockwise **quantize / dequantize** run
  on hand-written Metal kernels, each verified bit-exact against the CPU reference.
- **Native 4-bit matmul:** `gemv_4bit` (the M=1 inference case) runs through a **fused**
  Metal kernel (dequant + dot product in registers, the dequantized weight matrix is never
  materialized) — measured **3.4–6.2x** over the previous dequant+`F.linear` path.
  `gemm_4bit` (general M) runs natively for **fp16/fp32** (Metal dequant into a scratch
  buffer + `MPSMatrixMultiplication`, one command buffer): ~**2.5x** at small M, ~1.5x at
  medium M, and **~parity with `F.linear` at large M** (the GEMM itself dominates there).
  **bf16 `gemm_4bit` falls back** to dequant+`F.linear` — `MPSMatrixMultiplication` has no
  bf16 support (verified on macOS 26.4.1). Numbers: `MPS_STATUS.md` §11.
- **Not supported on `mps`:** LLM.int8() and the 8-bit optimizers (see the matrix below).
- **Graceful fallback:** if the native library is not present, the `mps` backend
  transparently uses a pure-PyTorch implementation — nothing hard-crashes.

## Requirements

| Requirement | Minimum                                                                   |
| ----------- | ------------------------------------------------------------------------- |
| macOS       | 14 (Sonoma)+                                                              |
| Hardware    | Apple Silicon (M1 or newer)                                               |
| PyTorch     | >= 2.4 with MPS (`torch.backends.mps.is_available()` → `True`)            |
| Build only  | CMake >= 3.31.6, Python >= 3.10, Xcode command line tools (`xcrun metal`) |

## Install / build

Native MPS is shipped inside the built wheel, so a normal install uses the Metal kernels
when they are packaged for your platform:

```bash
pip install bitsandbytes
```

To build from source (development, or a platform without a prebuilt wheel):

```bash
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git && cd bitsandbytes/
cmake -DCOMPUTE_BACKEND=mps -S .          # in-source: metallib lands next to the dylib
cmake --build . --config Release          # -> bitsandbytes/libbitsandbytes_mps.dylib + bitsandbytes.metallib
pip install -e .
```

Build it in-source (`-S .`, build dir at the repo root) so the compiled
`bitsandbytes.metallib` lands next to `libbitsandbytes_mps.dylib` in the `bitsandbytes/`
package directory, where the loader (via `dladdr`) expects it. Set `BNB_MPS_METALLIB` to
override the metallib path if needed.

## Supported-op matrix (`mps`)

| Operation                                            | On `mps`              | Notes                                                                                            |
| ---------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------------------ |
| `quantize_blockwise` (8-bit)                         | ✅ native Metal       | bit-exact vs CPU; pure-torch fallback                                                            |
| `dequantize_blockwise` (8-bit)                       | ✅ native Metal       | bit-exact vs CPU; pure-torch fallback                                                            |
| `quantize_4bit` (NF4/FP4)                            | ✅ native Metal       | packed nibbles + absmax bit-exact; fallback                                                      |
| `dequantize_4bit` (NF4/FP4)                          | ✅ native Metal       | bit-exact vs CPU; fallback                                                                       |
| `gemv_4bit` (M=1 inference)                          | ✅ native Metal       | fused dequant+dot kernel, 3.4–6.2x vs dequant+linear; fallback                                   |
| `gemm_4bit` (general M)                              | ✅ native (fp16/fp32) | dequant + `MPSMatrixMultiplication`; **bf16 falls back** to dequant+linear; large-M ≈ `F.linear` |
| LLM.int8() (`int8_*` ops)                            | ❌ not supported      | `int8_double_quant` raises `NotImplementedError`; no native int8 path                            |
| 8-bit optimizers (`optimizer_update_8bit_blockwise`) | ❌ not supported      | raises `NotImplementedError` on `mps`                                                            |

Legend: ✅ native Metal kernel · 〰️ functional but unfused/unoptimized · ❌ not supported.

### Native vs fallback vs unsupported

- **Native (Metal):** the four quant/dequant ops plus the two 4-bit matmuls above. When
  `libbitsandbytes_mps.dylib` + `bitsandbytes.metallib` are present and the load-time
  buffer-contract check passes, these dispatch to hand-written Metal kernels (the fp16/fp32
  `gemm_4bit` additionally routes through `MPSMatrixMultiplication`).
- **Fallback (pure-PyTorch):** any native op automatically falls back to a pure-PyTorch
  implementation when the native library is absent (unbuilt source checkout, or a wheel
  without the artifacts), and the matmuls also fall back for shapes/dtypes the native path
  does not accept (bf16 `gemm_4bit`, K not a multiple of 32, non-power-of-two blocksize).
- **Unsupported:** LLM.int8() and the 8-bit optimizers. Do not expect these on `mps` yet.

## Numerics & correctness

Correctness is validated **CPU-as-oracle** (there is no CUDA on a Mac): every native kernel
is compared against the `default` pure-PyTorch implementation. Quantized/packed outputs
(codes, packed nibbles, absmax) are asserted **bit-exact**; float dequant outputs use
documented per-dtype tolerances (and measure bit-exact in practice). The parity harness is
`tests/test_mps_parity.py`; run it on an Apple Silicon Mac with:

```bash
pytest tests/test_mps_parity.py -v
# require the native path (fail if it did not load), e.g. to verify a source build:
BNB_MPS_REQUIRE_NATIVE=1 pytest tests/test_mps_parity.py -v
```

## Known limitations

- **bf16 `gemm_4bit` is not native** — `MPSMatrixMultiplication` supports only
  fp32/fp16/int8/int16 (verified on macOS 26.4.1), so bf16 batched matmul uses the
  dequant + `F.linear` fallback. (bf16 `gemv_4bit`, the inference case, IS native/fused.)
- **Large-M `gemm_4bit` is ~parity, not a win** — at M ≳ 2048 the GEMM itself dominates
  and `MPSMatrixMultiplication` ≈ `F.linear`'s own GEMM. The native win is small/medium M
  and gemv.
- **A fixed ~0.15 ms sync tax per native call** — the native kernels run on their own
  Metal command queue, so each call pays a command-buffer round trip plus a
  `torch.mps.synchronize()`. torch exposes no safe handle to its own MPS stream, so this
  is a documented standing cost (measured breakdown and the full investigation:
  `MPS_STATUS.md` §11.3). It dominates only the smallest calls.
- **Per-call input copy for non-fresh views** — native ops clone any input with
  `storage_offset != 0` (a view's `data_ptr()` is base+offset, not a Metal buffer —
  verified, see `MPS_STATUS.md` §11.4). Steady-state matmul inputs are offset-0, so this
  rarely fires.
- **LLM.int8() and 8-bit optimizers** are not implemented on `mps`.

## Further reading

- [`PORT_PLAN.md`](./PORT_PLAN.md) — the full phased implementation spec.
- [`MPS_STATUS.md`](./MPS_STATUS.md) — ground-truth per-op audit, tolerances, and the
  packaging/verification record.
- [`NEXT_MATMUL_PLAN.md`](./NEXT_MATMUL_PLAN.md) — the executable spec for the native 4-bit
  matmul phase (completed in Phases M1–M4; results in `MPS_STATUS.md` §10–§11).
