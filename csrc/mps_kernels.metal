#include <metal_stdlib>
using namespace metal;

// Blockwise 8-bit quantization, hand-written to match the CPU/default reference in
// bitsandbytes/backends/default/ops.py::quantize_blockwise bit-for-bit:
//
//   absmax[b] = max(|A[i]|) over the block
//   scaled    = clamp(A[i] * (1 / max(absmax[b], 1e-38)), -1, 1)
//   out[i]    = searchsorted_left(scaled, bounds), bounds[j] = (code[j] + code[j+1]) / 2
//
// `code` is the 256-entry, sorted quantization map (same assumption as the CUDA kernel).
// `searchsorted_left` reproduces torch.bucketize(..., right=False): the number of midpoint
// bounds strictly less than `scaled`, yielding an index in [0, 255].
//
// One thread per block. This is the correctness-first shape (no SIMD-group reduction yet);
// a per-block parallel absmax reduction is a later perf phase. The metallib is compiled
// with -fno-fast-math (see CMakeLists.txt) so division is correctly rounded and no FMA
// contraction occurs, which is what keeps the bucket selection identical to the CPU oracle.

kernel void quantize_blockwise(
    device const float* code [[buffer(0)]],
    device const float* A [[buffer(1)]],
    device uchar* out [[buffer(2)]],
    device float* absmax [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    constant uint& blocksize [[buffer(5)]],
    uint block_id [[thread_position_in_grid]]
) {
    const uint start = block_id * blocksize;
    if (start >= n) {
        return;
    }
    const uint end = min(start + blocksize, n);

    // Per-block absmax (serial reduction over this block's elements).
    float amax = 0.0f;
    for (uint i = start; i < end; ++i) {
        amax = fmax(amax, fabs(A[i]));
    }
    absmax[block_id] = amax;

    // Match the reference's reciprocal-then-multiply (not a direct divide).
    const float inv = 1.0f / fmax(amax, 1e-38f);

    for (uint i = start; i < end; ++i) {
        const float scaled = clamp(A[i] * inv, -1.0f, 1.0f);

        // searchsorted-left over the 255 midpoint bounds of the 256-entry code table.
        uint lo = 0;
        uint hi = 255;
        while (lo < hi) {
            const uint mid = (lo + hi) >> 1;
            const float bound = (code[mid] + code[mid + 1]) * 0.5f;
            if (bound < scaled) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[i] = (uchar)lo;
    }
}
