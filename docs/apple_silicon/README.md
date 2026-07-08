# bitsandbytes on Apple Silicon (Metal / `mps`)

Preview support for Apple Silicon GPUs via native Metal kernels behind the PyTorch `mps`
backend. This page is the user/developer-facing summary; the executable spec and the
ground-truth parity audit live alongside it (see [Further reading](#further-reading)).

## Status at a glance

- **Native, bit-exact:** 4-bit (NF4/FP4) and 8-bit blockwise **quantize / dequantize** run
  on hand-written Metal kernels, each verified bit-exact against the CPU reference.
- **Works but unfused:** 4-bit (QLoRA-style) **inference** runs end to end, but the 4-bit
  matmul is **not fused** — quantized weights are dequantized through a Metal kernel and the
  matmul is then `torch.nn.functional.linear`. Correct, not yet performance-optimized.
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

| Operation                                            | On `mps`              | Notes                                                                     |
| ---------------------------------------------------- | --------------------- | ------------------------------------------------------------------------- |
| `quantize_blockwise` (8-bit)                         | ✅ native Metal       | bit-exact vs CPU; pure-torch fallback                                     |
| `dequantize_blockwise` (8-bit)                       | ✅ native Metal       | bit-exact vs CPU; pure-torch fallback                                     |
| `quantize_4bit` (NF4/FP4)                            | ✅ native Metal       | packed nibbles + absmax bit-exact; fallback                               |
| `dequantize_4bit` (NF4/FP4)                          | ✅ native Metal       | bit-exact vs CPU; fallback                                                |
| `gemv_4bit` / `gemm_4bit`                            | 〰️ works, **unfused** | native dequant + `torch.nn.functional.linear` (not a fused matmul kernel) |
| LLM.int8() (`int8_*` ops)                            | ❌ not supported      | `int8_double_quant` raises `NotImplementedError`; no native int8 path     |
| 8-bit optimizers (`optimizer_update_8bit_blockwise`) | ❌ not supported      | raises `NotImplementedError` on `mps`                                     |

Legend: ✅ native Metal kernel · 〰️ functional but unfused/unoptimized · ❌ not supported.

### Native vs fallback vs unsupported

- **Native (Metal):** the four quant/dequant ops above. When
  `libbitsandbytes_mps.dylib` + `bitsandbytes.metallib` are present and the load-time
  buffer-contract check passes, these dispatch to hand-written Metal kernels.
- **Fallback (pure-PyTorch):** any native op automatically falls back to a pure-PyTorch
  implementation when the native library is absent (unbuilt source checkout, or a wheel
  without the artifacts). The 4-bit matmuls are fallback-shaped by design (dequant + linear).
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

- **4-bit matmul is not fused** — dequantize-through-Metal + `F.linear`. A fused
  `gemv_4bit`/`gemm_4bit` kernel is the next planned work.
- **Per-call input copy** — native ops force inputs to fresh, offset-0 buffers before
  dispatch (one copy per call). Correctness-first; an optimization target.
- **LLM.int8() and 8-bit optimizers** are not implemented on `mps`.

## Further reading

- [`PORT_PLAN.md`](./PORT_PLAN.md) — the full phased implementation spec.
- [`MPS_STATUS.md`](./MPS_STATUS.md) — ground-truth per-op audit, tolerances, and the
  packaging/verification record.
- [`NEXT_MATMUL_PLAN.md`](./NEXT_MATMUL_PLAN.md) — the executable spec for the next phase:
  native 4-bit matmul fusion.
