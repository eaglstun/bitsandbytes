"""CPU-as-oracle parity tests for the MPS backend (Phase 1 of the Apple Silicon Metal port).

For every op the ``mps`` backend currently supports, these tests run the same seeded
inputs through the CPU path (on a source checkout without a native build, the ``cpu``
device resolves to the ``default`` pure-PyTorch backend -- the oracle) and through the
``mps`` path, then assert agreement within documented per-dtype tolerances.

Tolerances (empirically calibrated on torch 2.12.1 / macOS 26.4.1, see
``docs/apple_silicon/MPS_STATUS.md`` for the measured baseline):

- Quantization artifacts (uint8 codes, packed nibbles) must be **bit-exact**: both
  paths share the same fp32 quantization math, and any mismatch means a wrong bucket,
  not a rounding difference.
- ``absmax`` and other fp32 statistics: tight fp32 tolerance.
- Matmul outputs (gemv_4bit / gemm_4bit / int8 matmuls): fp32 tight; fp16/bf16 looser,
  following the per-dtype convention used for CUDA (fp32 1e-5, fp16 1e-2, bf16 4e-2
  absolute), so the same bounds keep working once native Metal kernels replace the
  pure-torch fallbacks in Phase 2+.

The whole module skips when MPS is not available.
"""

import pytest
import torch

import bitsandbytes
import bitsandbytes.functional as F
from tests.helpers import describe_dtype, id_formatter

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")

FLOAT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
BLOCKSIZES = [64, 128, 256, 512]

# Per-dtype (rtol, atol) for comparing MPS results against the CPU oracle.
# fp32 divergence comes only from accumulation order (measured <= ~8e-6 at K<=256);
# fp16/bf16 get the looser bounds to absorb half-precision rounding.
PARITY_TOLERANCE = {
    torch.float32: (1e-6, 1e-5),
    torch.float16: (1e-3, 1e-2),
    torch.bfloat16: (1e-2, 4e-2),
}


def assert_parity(res_mps: torch.Tensor, res_cpu: torch.Tensor, dtype: torch.dtype):
    """Assert an MPS result matches the CPU oracle within the documented tolerance."""
    assert res_mps.device.type == "mps"
    rtol, atol = PARITY_TOLERANCE[dtype]
    torch.testing.assert_close(res_mps.cpu(), res_cpu, rtol=rtol, atol=atol)


def assert_bit_exact(res_mps: torch.Tensor, res_cpu: torch.Tensor):
    """Quantized codes must match bucket-for-bucket, not just approximately."""
    assert res_mps.device.type == "mps"
    if res_cpu.dtype != torch.uint8:
        res_cpu = res_cpu.view(torch.uint8)
        res_mps = res_mps.view(torch.uint8)
    mismatched = (res_mps.cpu() != res_cpu).sum().item()
    assert mismatched == 0, f"{mismatched}/{res_cpu.numel()} quantized values differ from CPU oracle"


class TestBlockwise8bitParity:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_quantize_blockwise(self, dtype, blocksize):
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_blockwise(A.to("mps"), code.to("mps"), blocksize)

        assert q_mps.shape == q_cpu.shape
        assert q_mps.dtype == torch.uint8
        assert_bit_exact(q_mps, q_cpu)
        assert_parity(absmax_mps, absmax_cpu, torch.float32)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_dequantize_blockwise(self, dtype, blocksize):
        # NOTE: there is no "mps" registration for dequantize_blockwise; this runs
        # through the "default" (pure-torch) kernel on mps tensors.
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)

        # Quantize once on CPU so both dequant paths see identical inputs.
        q, absmax = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)

        dq_cpu = torch.ops.bitsandbytes.dequantize_blockwise(q, absmax, code, blocksize, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_blockwise(
            q.to("mps"), absmax.to("mps"), code.to("mps"), blocksize, dtype
        )

        assert dq_mps.shape == A.shape
        assert dq_mps.dtype == dtype
        assert_parity(dq_mps, dq_cpu, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_roundtrip_reconstruction(self, dtype, blocksize):
        """quantize->dequantize entirely on MPS reconstructs as well as the CPU oracle."""
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)
        code = F.create_dynamic_map().to(torch.float32)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_blockwise(A, code, blocksize)
        dq_cpu = torch.ops.bitsandbytes.dequantize_blockwise(q_cpu, absmax_cpu, code, blocksize, dtype)

        A_mps = A.to("mps")
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_blockwise(A_mps, code.to("mps"), blocksize)
        dq_mps = torch.ops.bitsandbytes.dequantize_blockwise(
            q_mps, absmax_mps, code.to("mps"), blocksize, dtype
        )

        err_cpu = (dq_cpu.float() - A.float()).abs().mean().item()
        err_mps = (dq_mps.cpu().float() - A.float()).abs().mean().item()

        # Dynamic 8-bit reconstruction of randn is ~1e-2 mean abs error; "confident
        # garbage" would be ~1.0. The MPS error must also track the oracle closely.
        assert err_mps < 0.05, f"MPS roundtrip error {err_mps} implausibly high"
        assert err_mps == pytest.approx(err_cpu, rel=0.02)


class Test4bitParity:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_quantize_4bit(self, dtype, quant_type, blocksize):
        torch.manual_seed(1337)
        A = torch.randn(256, 256, dtype=dtype)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(A.to("mps"), blocksize, quant_type, torch.uint8)

        assert q_mps.shape == q_cpu.shape
        assert q_mps.dtype == torch.uint8
        assert_bit_exact(q_mps, q_cpu)
        assert_parity(absmax_mps, absmax_cpu, torch.float32)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_dequantize_4bit(self, dtype, quant_type, blocksize):
        torch.manual_seed(1337)
        shape = (256, 256)
        A = torch.randn(shape, dtype=dtype)

        # Quantize once on CPU so both dequant paths see identical inputs.
        q, absmax = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)

        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q, absmax, blocksize, quant_type, shape, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(
            q.to("mps"), absmax.to("mps"), blocksize, quant_type, shape, dtype
        )

        assert dq_mps.shape == shape
        assert dq_mps.dtype == dtype
        assert_parity(dq_mps, dq_cpu, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", BLOCKSIZES)
    def test_roundtrip_reconstruction(self, dtype, quant_type, blocksize):
        """NF4/FP4 quantize->dequantize entirely on MPS reconstructs like the CPU oracle."""
        torch.manual_seed(1337)
        shape = (256, 256)
        A = torch.randn(shape, dtype=dtype)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)
        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q_cpu, absmax_cpu, blocksize, quant_type, shape, dtype)

        A_mps = A.to("mps")
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(A_mps, blocksize, quant_type, torch.uint8)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(q_mps, absmax_mps, blocksize, quant_type, shape, dtype)

        err_cpu = (dq_cpu.float() - A.float()).abs().mean().item()
        err_mps = (dq_mps.cpu().float() - A.float()).abs().mean().item()

        # 4-bit reconstruction of randn is ~6e-2 mean abs error; garbage would be ~1.0.
        assert err_mps < 0.15, f"MPS roundtrip error {err_mps} implausibly high"
        assert err_mps == pytest.approx(err_cpu, rel=0.02)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", [64, 128, 256])
    def test_roundtrip_partial_block(self, dtype, quant_type, blocksize):
        """Roundtrip parity when numel is not divisible by blocksize (tail block path)."""
        torch.manual_seed(1337)
        shape = (7, blocksize - 1)
        A = torch.randn(shape, dtype=dtype)

        q_cpu, absmax_cpu = torch.ops.bitsandbytes.quantize_4bit(A, blocksize, quant_type, torch.uint8)
        q_mps, absmax_mps = torch.ops.bitsandbytes.quantize_4bit(A.to("mps"), blocksize, quant_type, torch.uint8)

        assert_bit_exact(q_mps, q_cpu)
        assert_parity(absmax_mps, absmax_cpu, torch.float32)

        dq_cpu = torch.ops.bitsandbytes.dequantize_4bit(q_cpu, absmax_cpu, blocksize, quant_type, shape, dtype)
        dq_mps = torch.ops.bitsandbytes.dequantize_4bit(q_mps, absmax_mps, blocksize, quant_type, shape, dtype)

        assert dq_mps.shape == shape
        assert torch.isfinite(dq_mps.cpu()).all()
        assert_parity(dq_mps, dq_cpu, dtype)


class TestMatmul4bitParity:
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("blocksize", [64, 256])
    def test_gemv_4bit(self, dtype, quant_type, blocksize):
        torch.manual_seed(1337)
        out_features, in_features = 1024, 256
        A = torch.randn(1, 1, in_features, dtype=dtype)
        B = torch.randn(out_features, in_features, dtype=dtype)

        # Quantize B once on CPU (quantization is bit-exact across devices).
        B_q, absmax = torch.ops.bitsandbytes.quantize_4bit(B, blocksize, quant_type, torch.uint8)
        code = F.get_4bit_type(quant_type, device="cpu", blocksize=blocksize)

        out_cpu = torch.ops.bitsandbytes.gemv_4bit(A, B_q, B.shape, absmax, code, blocksize)
        out_mps = torch.ops.bitsandbytes.gemv_4bit(
            A.to("mps"), B_q.to("mps"), B.shape, absmax.to("mps"), code.to("mps"), blocksize
        )

        assert out_mps.shape == (1, 1, out_features)
        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES, ids=describe_dtype)
    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    @pytest.mark.parametrize("compress_statistics", [False, True], ids=id_formatter("compress_statistics"))
    @pytest.mark.parametrize("has_bias", [False, True], ids=id_formatter("has_bias"))
    def test_gemm_4bit(self, dtype, quant_type, compress_statistics, has_bias):
        torch.manual_seed(1337)
        N, K, blocksize = 64, 64, 64
        A = torch.randn(2, 2, K, dtype=dtype)
        B = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=dtype) if has_bias else None

        # Quantize on each device via the public API; parity of the quantization
        # itself is covered by Test4bitParity.
        B_q, qs = bitsandbytes.functional.quantize_4bit(
            B, blocksize=blocksize, quant_type=quant_type, compress_statistics=compress_statistics
        )
        B_q_mps, qs_mps = bitsandbytes.functional.quantize_4bit(
            B.to("mps"), blocksize=blocksize, quant_type=quant_type, compress_statistics=compress_statistics
        )

        if compress_statistics:
            out_cpu = torch.ops.bitsandbytes.gemm_4bit(
                A,
                B_q,
                list(B.shape),
                qs.state2.absmax,
                blocksize,
                quant_type,
                bias=bias,
                absmax_8bit=qs.absmax,
                absmax_code=qs.state2.code,
                absmax_offset=qs.offset,
            )
            out_mps = torch.ops.bitsandbytes.gemm_4bit(
                A.to("mps"),
                B_q_mps,
                list(B.shape),
                qs_mps.state2.absmax,
                blocksize,
                quant_type,
                bias=bias.to("mps") if bias is not None else None,
                absmax_8bit=qs_mps.absmax,
                absmax_code=qs_mps.state2.code,
                absmax_offset=qs_mps.offset,
            )
        else:
            out_cpu = torch.ops.bitsandbytes.gemm_4bit(A, B_q, list(B.shape), qs.absmax, blocksize, quant_type, bias=bias)
            out_mps = torch.ops.bitsandbytes.gemm_4bit(
                A.to("mps"),
                B_q_mps,
                list(B.shape),
                qs_mps.absmax,
                blocksize,
                quant_type,
                bias=bias.to("mps") if bias is not None else None,
            )

        assert out_mps.shape == (2, 2, N)
        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)


class TestInt8Parity:
    """LLM.int8() ops on mps all resolve to the "default" (pure-torch) kernels."""

    def test_int8_linear_matmul(self):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8)

        out_cpu = torch.ops.bitsandbytes.int8_linear_matmul(A, B)
        out_mps = torch.ops.bitsandbytes.int8_linear_matmul(A.to("mps"), B.to("mps"))

        assert out_mps.dtype == torch.int32
        # int32 accumulations of int8 products are exactly representable in fp32
        # at these sizes; results must match exactly.
        assert torch.equal(out_mps.cpu(), out_cpu)

    def test_int8_linear_matmul_out(self):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8)

        out_cpu = torch.empty((10, 30), dtype=torch.int32)
        torch.ops.bitsandbytes.int8_linear_matmul.out(A, B, out_cpu)

        out_mps = torch.empty((10, 30), dtype=torch.int32, device="mps")
        torch.ops.bitsandbytes.int8_linear_matmul.out(A.to("mps"), B.to("mps"), out_mps)

        assert torch.equal(out_mps.cpu(), out_cpu)

    @pytest.mark.parametrize("threshold", [0.0, 6.0])
    def test_int8_vectorwise_quant(self, threshold):
        torch.manual_seed(1337)
        A = torch.randn(10, 20, dtype=torch.float16)
        A[1][0] = 1000.0  # outlier

        q_cpu, stats_cpu, outliers_cpu = torch.ops.bitsandbytes.int8_vectorwise_quant(A.clone(), threshold=threshold)
        q_mps, stats_mps, outliers_mps = torch.ops.bitsandbytes.int8_vectorwise_quant(
            A.clone().to("mps"), threshold=threshold
        )

        assert torch.equal(q_mps.cpu(), q_cpu)
        assert_parity(stats_mps, stats_cpu, torch.float32)
        if threshold > 0.0:
            assert outliers_mps is not None
            assert torch.equal(outliers_mps.cpu(), outliers_cpu)
        else:
            assert outliers_mps is None

    def test_int8_vectorwise_dequant(self):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        stats = torch.rand(10, dtype=torch.float32) * 5

        out_cpu = torch.ops.bitsandbytes.int8_vectorwise_dequant(A, stats)
        out_mps = torch.ops.bitsandbytes.int8_vectorwise_dequant(A.to("mps"), stats.to("mps"))

        assert_parity(out_mps, out_cpu, torch.float32)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    def test_int8_mm_dequant(self, dtype):
        torch.manual_seed(1337)
        A = torch.randint(-1000, 1000, (32, 32), dtype=torch.int32)
        row_stats = torch.rand(32, dtype=torch.float32) * 3
        col_stats = torch.rand(32, dtype=torch.float32) * 3

        out_cpu = torch.ops.bitsandbytes.int8_mm_dequant(A, row_stats, col_stats, dtype=dtype)
        out_mps = torch.ops.bitsandbytes.int8_mm_dequant(
            A.to("mps"), row_stats.to("mps"), col_stats.to("mps"), dtype=dtype
        )

        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=describe_dtype)
    @pytest.mark.parametrize("has_bias", [False, True], ids=id_formatter("has_bias"))
    def test_int8_scaled_mm(self, dtype, has_bias):
        torch.manual_seed(1337)
        A = torch.randint(-128, 127, (10, 20), dtype=torch.int8)
        B = torch.randint(-128, 127, (30, 20), dtype=torch.int8)
        row_stats = torch.rand(10, dtype=torch.float32)
        col_stats = torch.rand(30, dtype=torch.float32)
        bias = torch.randn(30, dtype=dtype) if has_bias else None

        out_cpu = torch.ops.bitsandbytes.int8_scaled_mm(A, B, row_stats, col_stats, bias=bias, dtype=dtype)
        out_mps = torch.ops.bitsandbytes.int8_scaled_mm(
            A.to("mps"),
            B.to("mps"),
            row_stats.to("mps"),
            col_stats.to("mps"),
            bias=bias.to("mps") if bias is not None else None,
            dtype=dtype,
        )

        assert out_mps.dtype == dtype
        assert_parity(out_mps, out_cpu, dtype)


def _run_optimizer_32bit_both_devices(optimizer_name: str, weight_decay: float, steps=(1, 2, 3)):
    torch.manual_seed(1337)
    g = torch.randn(256, dtype=torch.float32)
    p = torch.randn(256, dtype=torch.float32)
    state1 = torch.zeros(256, dtype=torch.float32)
    state2 = torch.zeros(256, dtype=torch.float32) if optimizer_name == "adam" else None

    g_mps = g.to("mps")
    p_mps = p.clone().to("mps")
    state1_mps = state1.clone().to("mps")
    state2_mps = state2.clone().to("mps") if state2 is not None else None

    for step in steps:
        args = (0.0, 0.0, 0.9, 0.999, 0.0, 0.0, 1e-8, weight_decay, step, 1e-3, 1.0)
        torch.ops.bitsandbytes.optimizer_update_32bit(optimizer_name, g, p, state1, state2, None, *args)
        torch.ops.bitsandbytes.optimizer_update_32bit(
            optimizer_name, g_mps, p_mps, state1_mps, state2_mps, None, *args
        )

    return (p, state1, state2), (p_mps, state1_mps, state2_mps)


class TestOptimizerParity:
    @pytest.mark.parametrize("optimizer_name", ["adam", "momentum", "rmsprop", "lion"])
    def test_optimizer_update_32bit(self, optimizer_name):
        # On mps this resolves to the "default" (pure-torch) kernel; the cpu oracle
        # runs the dedicated "cpu" kernel from backends/cpu/ops.py.
        # NOTE: lion is tested with weight_decay=0.0 because the two backends
        # disagree on lion weight-decay semantics -- see
        # test_lion_weight_decay_backend_divergence below.
        weight_decay = 0.0 if optimizer_name == "lion" else 0.01
        (p, state1, state2), (p_mps, state1_mps, state2_mps) = _run_optimizer_32bit_both_devices(
            optimizer_name, weight_decay
        )

        assert_parity(p_mps, p, torch.float32)
        assert_parity(state1_mps, state1, torch.float32)
        if state2 is not None:
            assert_parity(state2_mps, state2, torch.float32)

    @pytest.mark.xfail(
        strict=True,
        reason="Known cross-backend divergence: the 'default' kernel (used on mps) applies COUPLED "
        "weight decay for lion (g += p*wd, backends/default/ops.py optimizer_id in [0,1,2,4]), while "
        "the 'cpu' kernel and the CUDA kernel apply DECOUPLED weight decay (p *= 1 - lr*wd), matching "
        "the Lion paper. The default backend is the outlier. See docs/apple_silicon/MPS_STATUS.md.",
    )
    def test_lion_weight_decay_backend_divergence(self):
        (p, state1, _), (p_mps, state1_mps, _) = _run_optimizer_32bit_both_devices("lion", weight_decay=0.01)

        assert_parity(p_mps, p, torch.float32)
        assert_parity(state1_mps, state1, torch.float32)


class TestKnownGapsOnMps:
    """Ops with no "mps" and no "default" registration must fail loudly on mps.

    These document the current coverage gaps (see docs/apple_silicon/MPS_STATUS.md).
    If one of these tests starts failing because the op now *works* on mps, an
    implementation has been registered: move the op into the parity tests above and
    update MPS_STATUS.md.
    """

    def test_dequantize_blockwise_out_missing(self):
        A = torch.randint(0, 256, (4096,), dtype=torch.uint8, device="mps")
        code = F.create_dynamic_map().to("mps", torch.float32)
        absmax = torch.rand(16, device="mps")
        out = torch.empty(4096, dtype=torch.float32, device="mps")

        with pytest.raises(NotImplementedError):
            torch.ops.bitsandbytes.dequantize_blockwise.out(A, absmax, code, 256, torch.float32, out)

    def test_int8_double_quant_missing(self):
        A = torch.randn(10, 20, dtype=torch.float16, device="mps")

        with pytest.raises(NotImplementedError):
            torch.ops.bitsandbytes.int8_double_quant(A)

    def test_optimizer_update_8bit_blockwise_missing(self):
        g = torch.randn(256, device="mps")
        p = torch.randn(256, device="mps")
        state1 = torch.zeros(256, dtype=torch.uint8, device="mps")
        qmap = F.create_dynamic_map(signed=True).to("mps")
        absmax = torch.zeros(1, device="mps")

        with pytest.raises(NotImplementedError):
            torch.ops.bitsandbytes.optimizer_update_8bit_blockwise(
                "adam", g, p, state1, None, 0.9, 0.999, 0.0, 0.0, 1e-8, 1, 1e-3, qmap, None, absmax, None, 0.0, 1.0
            )
