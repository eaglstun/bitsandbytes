#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <cstdint>
#include <dlfcn.h>
#include <libgen.h>
#include <mach/mach_time.h>
#include <string>

// Native Metal dispatch layer for the bitsandbytes MPS backend.
//
// Buffer bridging: torch MPS tensors store their id<MTLBuffer> as the storage data
// pointer, so tensor.data_ptr() (passed here from Python via ctypes as a void*) IS the
// id<MTLBuffer>. We cast it directly -- no libtorch linkage required. The caller
// guarantees each tensor is contiguous with storage_offset == 0 (a fresh allocation),
// so binding the buffer at offset 0 is correct.
//
// Synchronization: this file dispatches on its own command queue, separate from torch's
// MPS stream. The Python caller therefore flushes torch's queue (torch.mps.synchronize())
// BEFORE calling in -- so the input buffers are materialized -- and we block on
// waitUntilCompleted AFTER commit, so the outputs are complete before Python (and torch)
// read them. Correctness-first: the blocking wait is intentional.

static id<MTLDevice> get_device() {
    static id<MTLDevice> device = nil;
    if (!device) {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"bitsandbytes: failed to get default Metal device");
            abort();
        }
    }
    return device;
}

static id<MTLCommandQueue> get_queue() {
    static id<MTLCommandQueue> queue = nil;
    if (!queue) {
        queue = [get_device() newCommandQueue];
        if (!queue) {
            NSLog(@"bitsandbytes: failed to create Metal command queue");
            abort();
        }
    }
    return queue;
}

// Resolve bitsandbytes.metallib next to THIS loaded dylib (install-safe), not by a
// CWD-relative path. Honors BNB_MPS_METALLIB as an override.
static NSString* metallib_path() {
    const char* override_path = getenv("BNB_MPS_METALLIB");
    if (override_path && override_path[0] != '\0') {
        return [NSString stringWithUTF8String:override_path];
    }

    Dl_info info;
    if (dladdr(reinterpret_cast<const void*>(&metallib_path), &info) && info.dli_fname) {
        std::string path(info.dli_fname);
        // dirname may mutate its argument; operate on a copy.
        std::string dir(path);
        char* d = dirname(&dir[0]);
        std::string metallib = std::string(d) + "/bitsandbytes.metallib";
        return [NSString stringWithUTF8String:metallib.c_str()];
    }

    // Last resort: CWD-relative (matches historical behavior).
    return @"bitsandbytes.metallib";
}

static id<MTLLibrary> get_library() {
    static id<MTLLibrary> library = nil;
    if (!library) {
        NSError* error = nil;
        NSString* path = metallib_path();
        library = [get_device() newLibraryWithURL:[NSURL fileURLWithPath:path] error:&error];
        if (!library) {
            NSLog(@"bitsandbytes: failed to load metallib at %@: %@", path, error);
            abort();
        }
    }
    return library;
}

static id<MTLComputePipelineState> get_pipeline(NSString* name) {
    static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* cache = nil;
    if (!cache) {
        cache = [[NSMutableDictionary alloc] init];
    }
    id<MTLComputePipelineState> pso = cache[name];
    if (pso) {
        return pso;
    }

    id<MTLFunction> fn = [get_library() newFunctionWithName:name];
    if (!fn) {
        NSLog(@"bitsandbytes: kernel function '%@' not found in metallib", name);
        abort();
    }
    NSError* error = nil;
    pso = [get_device() newComputePipelineStateWithFunction:fn error:&error];
    if (!pso) {
        NSLog(@"bitsandbytes: failed to build pipeline for '%@': %@", name, error);
        abort();
    }
    cache[name] = pso;
    return pso;
}

// Load-time guard for the data_ptr()-is-the-MTLBuffer contract (an undocumented torch
// internal). Given a pointer that Python obtained from a real MPS tensor's data_ptr() plus
// that tensor's byte size, verify it resolves to a genuine id<MTLBuffer> of at least that
// size. Returns 1 on success, 0 if the contract does not hold -- so a future torch that
// changes the meaning of data_ptr() surfaces as a clear, actionable failure (native path
// disabled + logged) instead of a blind cast of garbage. Cheap: called once at load.
extern "C" int bnb_mps_check_buffer_contract(void* ptr, int64_t min_bytes) {
    @autoreleasepool {
        if (!ptr) {
            return 0;
        }
        @try {
            id obj = (__bridge id)ptr;
            if (![obj conformsToProtocol:@protocol(MTLBuffer)]) {
                return 0;
            }
            id<MTLBuffer> buf = (id<MTLBuffer>)obj;
            if ((int64_t)[buf length] < min_bytes) {
                return 0;
            }
            return 1;
        } @catch (...) {
            return 0;
        }
    }
}

// Host time in seconds on the mach_absolute_time timebase -- the same clock
// MTLCommandBuffer's GPUStartTime/GPUEndTime report, so the two are directly comparable.
// Used only by the BNB_MPS_PROFILE probe.
static double host_time_s() {
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) {
        mach_timebase_info(&tb);
    }
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

// One thread per block: cap the threadgroup and dispatch a non-uniform grid, then block on
// completion. dispatchThreads is supported on all Apple Silicon GPUs.
static void dispatch_per_block(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, int64_t num_blocks) {
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
    if (tg > 256) {
        tg = 256;
    }
    if (tg > (NSUInteger)num_blocks) {
        tg = (NSUInteger)num_blocks;
    }
    if (tg == 0) {
        tg = 1;
    }
    [enc dispatchThreads:MTLSizeMake((NSUInteger)num_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// quantize_blockwise: code (float32[256]), A (float32[n]) -> out (uint8[n]), absmax
// (float32[ceil(n/blocksize)]). All pointers are torch MPS tensor data_ptr() values,
// i.e. id<MTLBuffer> objects for offset-0 contiguous tensors.
extern "C" void bnb_mps_quantize_blockwise(void* code, void* A, void* out, void* absmax, int64_t n, int64_t blocksize) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = get_pipeline(@"quantize_blockwise");
        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:(__bridge id<MTLBuffer>)code offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)absmax offset:0 atIndex:3];

        uint32_t n32 = (uint32_t)n;
        uint32_t bs32 = (uint32_t)blocksize;
        [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
        [enc setBytes:&bs32 length:sizeof(bs32) atIndex:5];

        dispatch_per_block(enc, pso, (n + blocksize - 1) / blocksize);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// dequantize_blockwise: code (float32[256]), A (uint8[n]), absmax (float32[blocks]) ->
// out (float32[n]). Python casts fp32 out to the requested dtype.
extern "C" void
    bnb_mps_dequantize_blockwise(void* code, void* A, void* absmax, void* out, int64_t n, int64_t blocksize) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = get_pipeline(@"dequantize_blockwise");
        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:(__bridge id<MTLBuffer>)code offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)absmax offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];

        uint32_t n32 = (uint32_t)n;
        uint32_t bs32 = (uint32_t)blocksize;
        [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
        [enc setBytes:&bs32 length:sizeof(bs32) atIndex:5];

        dispatch_per_block(enc, pso, (n + blocksize - 1) / blocksize);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// dequantize_4bit: code (float32[16]), A (uint8 packed), absmax (float32[blocks]) ->
// out (float32[n]). Python casts fp32 out to the requested dtype and reshapes.
extern "C" void bnb_mps_dequantize_4bit(void* code, void* A, void* absmax, void* out, int64_t n, int64_t blocksize) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = get_pipeline(@"dequantize_4bit");
        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:(__bridge id<MTLBuffer>)code offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)absmax offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];

        uint32_t n32 = (uint32_t)n;
        uint32_t bs32 = (uint32_t)blocksize;
        [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
        [enc setBytes:&bs32 length:sizeof(bs32) atIndex:5];

        dispatch_per_block(enc, pso, (n + blocksize - 1) / blocksize);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// gemv_4bit (fused dequant + matrix-vector multiply): code (float32[16]), B (uint8 packed,
// N*K/2 bytes), absmax (float32[blocks]), A (K elements of the activation dtype) ->
// out (N elements of the activation dtype). One threadgroup of 32 threads (one SIMD-group)
// per output element. The Python caller guarantees K % 32 == 0 and passes
// bs_shift = log2(blocksize); dtype_flag (0 = fp32, 1 = fp16, 2 = bf16) selects the kernel
// variant, so A and out bind directly in torch's dtype (no cast kernels on the torch queue).
extern "C" void bnb_mps_gemv_4bit(
    void* code, void* B, void* absmax, void* A, void* out, int64_t K, int64_t N, int64_t bs_shift, int64_t dtype_flag
) {
    @autoreleasepool {
        NSString* name = @"gemv_4bit_fp32";
        if (dtype_flag == 1) {
            name = @"gemv_4bit_fp16";
        } else if (dtype_flag == 2) {
            name = @"gemv_4bit_bf16";
        }
        id<MTLComputePipelineState> pso = get_pipeline(name);
        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:(__bridge id<MTLBuffer>)code offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)absmax offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:4];

        uint32_t K32 = (uint32_t)K;
        uint32_t shift32 = (uint32_t)bs_shift;
        [enc setBytes:&K32 length:sizeof(K32) atIndex:5];
        [enc setBytes:&shift32 length:sizeof(shift32) atIndex:6];

        // One SIMD-group per output element n.
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)N, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];

        // Timing probe (BNB_MPS_PROFILE=1): decomposes the blocking call into
        //   sched = commit -> GPU start (driver/queue scheduling latency)
        //   gpu   = kernel execution (GPUStartTime..GPUEndTime)
        //   done  = GPU end -> waitUntilCompleted return (completion delivery)
        // Wall-clock around the whole ctypes call additionally includes encode (above) and
        // the caller's torch.mps.synchronize().
        static const bool profile = getenv("BNB_MPS_PROFILE") != nullptr;
        const double t_commit = profile ? host_time_s() : 0.0;
        [cb commit];
        [cb waitUntilCompleted];
        if (profile) {
            const double t_done = host_time_s();
            const double sched_ms = ([cb GPUStartTime] - t_commit) * 1000.0;
            const double gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
            const double done_ms = (t_done - [cb GPUEndTime]) * 1000.0;
            NSLog(
                @"bnb_mps_gemv_4bit %@ N=%lld K=%lld sched=%.3fms gpu=%.3fms done=%.3fms", name, (long long)N,
                (long long)K, sched_ms, gpu_ms, done_ms
            );
        }
    }
}

// Growable scratch MTLBuffer for gemm_4bit's dequantized B. Private storage (GPU-only) --
// the CPU never touches B_dq. Safe to reuse a single static buffer because every entry
// point blocks on waitUntilCompleted before returning, so no two dispatches overlap.
// (This file is not thread-safe, matching the existing static caches.)
static id<MTLBuffer> get_scratch(size_t bytes) {
    static id<MTLBuffer> scratch = nil;
    if (!scratch || [scratch length] < bytes) {
        [scratch release];
        scratch = [get_device() newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (!scratch) {
            NSLog(@"bitsandbytes: failed to allocate %zu-byte GEMM scratch buffer", bytes);
            abort();
        }
    }
    return scratch;
}

// Shape-keyed MPSMatrixMultiplication cache. Operands are supplied at encode time, so one
// object per {M, N, K, dtype} can be reused across calls (transformer workloads repeat a
// few shapes, so the hit rate is high -- the CT2 Metal backend lesson).
static MPSMatrixMultiplication* get_gemm(int64_t M, int64_t N, int64_t K, int64_t dtype_flag) {
    static NSMutableDictionary<NSString*, MPSMatrixMultiplication*>* cache = nil;
    if (!cache) {
        cache = [[NSMutableDictionary alloc] init];
    }
    NSString* key = [NSString
        stringWithFormat:@"%lld_%lld_%lld_%lld", (long long)M, (long long)N, (long long)K, (long long)dtype_flag];
    MPSMatrixMultiplication* mm = cache[key];
    if (mm) {
        return mm;
    }
    // C[M, N] = A[M, K] . B_dq[N, K]^T  (row-major on both sides; MPS is row-major, so no
    // cuBLAS-style operand swap).
    mm = [[MPSMatrixMultiplication alloc] initWithDevice:get_device()
                                           transposeLeft:NO
                                          transposeRight:YES
                                              resultRows:(NSUInteger)M
                                           resultColumns:(NSUInteger)N
                                         interiorColumns:(NSUInteger)K
                                                   alpha:1.0
                                                    beta:0.0];
    if (!mm) {
        NSLog(
            @"bitsandbytes: failed to create MPSMatrixMultiplication (M=%lld N=%lld K=%lld)", (long long)M,
            (long long)N, (long long)K
        );
        abort();
    }
    cache[key] = mm;
    [mm release]; // the cache retains it
    return mm;
}

// gemm_4bit (general M): dequantize packed B into a scratch buffer in the activation dtype,
// then run MPSMatrixMultiplication A[M,K] . B_dq[N,K]^T -> out[M,N], plus an optional bias
// epilogue -- ALL encoded on ONE command buffer with ONE commit + ONE blocking wait. That
// single-sync structure is the point: the Phase-M2 finding is that the per-call cross-queue
// sync (~0.15-0.25ms) dominates wall-clock, so dequant-then-torch-F.linear pays it twice
// (native dequant wait + torch's own GEMM sync) while this path pays it once.
//
// code (float32[16]), B (uint8 packed, N*K/2 bytes), absmax (float32[N*K >> bs_shift]),
// A (M*K elements of T), bias (N elements of T, may be NULL), out (M*N elements of T).
// dtype_flag: 0 = fp32, 1 = fp16. bf16 is NOT accepted: MPSMatrixMultiplication asserts on
// anything but fp32/fp16/int8/int16 (verified on macOS 26.4.1), so the Python router keeps
// bf16 on the dequant + F.linear fallback.
extern "C" void bnb_mps_gemm_4bit(
    void* code, void* B, void* absmax, void* A, void* bias, void* out, int64_t M, int64_t K, int64_t N,
    int64_t bs_shift, int64_t dtype_flag
) {
    @autoreleasepool {
        const bool fp16 = (dtype_flag == 1);
        const size_t elsize = fp16 ? 2 : 4;
        const MPSDataType mps_dtype = fp16 ? MPSDataTypeFloat16 : MPSDataTypeFloat32;

        id<MTLBuffer> scratch = get_scratch((size_t)N * (size_t)K * elsize);
        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];

        // 1) Dequantize packed B -> scratch B_dq[N, K] in T (one thread per 32-element chunk).
        {
            id<MTLComputePipelineState> pso =
                get_pipeline(fp16 ? @"dequantize_4bit_chunked_fp16" : @"dequantize_4bit_chunked_fp32");
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:(__bridge id<MTLBuffer>)code offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)B offset:0 atIndex:1];
            [enc setBuffer:(__bridge id<MTLBuffer>)absmax offset:0 atIndex:2];
            [enc setBuffer:scratch offset:0 atIndex:3];
            uint32_t shift32 = (uint32_t)bs_shift;
            [enc setBytes:&shift32 length:sizeof(shift32) atIndex:4];

            const NSUInteger chunks = (NSUInteger)((N * K) >> 5); // K % 32 == 0 (router guard)
            NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
            if (tg > 256) {
                tg = 256;
            }
            if (tg > chunks) {
                tg = chunks;
            }
            if (tg == 0) {
                tg = 1;
            }
            [enc dispatchThreads:MTLSizeMake(chunks, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        // 2) GEMM on the same command buffer. Metal's automatic hazard tracking orders the
        //    MPS encoder after the dequant encoder (scratch is a tracked resource).
        {
            MPSMatrixDescriptor* dA = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                                                            columns:(NSUInteger)K
                                                                           rowBytes:(NSUInteger)K * elsize
                                                                           dataType:mps_dtype];
            MPSMatrixDescriptor* dB = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)N
                                                                            columns:(NSUInteger)K
                                                                           rowBytes:(NSUInteger)K * elsize
                                                                           dataType:mps_dtype];
            MPSMatrixDescriptor* dC = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)M
                                                                            columns:(NSUInteger)N
                                                                           rowBytes:(NSUInteger)N * elsize
                                                                           dataType:mps_dtype];
            MPSMatrix* mA = [[[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)A descriptor:dA] autorelease];
            MPSMatrix* mB = [[[MPSMatrix alloc] initWithBuffer:scratch descriptor:dB] autorelease];
            MPSMatrix* mC = [[[MPSMatrix alloc] initWithBuffer:(__bridge id<MTLBuffer>)out descriptor:dC] autorelease];
            [get_gemm(M, N, K, dtype_flag) encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        }

        // 3) Optional bias epilogue: out[m, n] += bias[n], still the same command buffer.
        if (bias) {
            id<MTLComputePipelineState> pso = get_pipeline(fp16 ? @"gemm_bias_add_fp16" : @"gemm_bias_add_fp32");
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:0];
            [enc setBuffer:(__bridge id<MTLBuffer>)bias offset:0 atIndex:1];
            uint32_t N32 = (uint32_t)N;
            [enc setBytes:&N32 length:sizeof(N32) atIndex:2];

            NSUInteger w = pso.threadExecutionWidth;
            NSUInteger h = pso.maxTotalThreadsPerThreadgroup / w;
            if (h > (NSUInteger)M) {
                h = (NSUInteger)M;
            }
            if (h == 0) {
                h = 1;
            }
            [enc dispatchThreads:MTLSizeMake((NSUInteger)N, (NSUInteger)M, 1)
                threadsPerThreadgroup:MTLSizeMake(w, h, 1)];
            [enc endEncoding];
        }

        static const bool profile = getenv("BNB_MPS_PROFILE") != nullptr;
        const double t_commit = profile ? host_time_s() : 0.0;
        [cb commit];
        [cb waitUntilCompleted];
        if (profile) {
            const double t_done = host_time_s();
            const double sched_ms = ([cb GPUStartTime] - t_commit) * 1000.0;
            const double gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
            const double done_ms = (t_done - [cb GPUEndTime]) * 1000.0;
            NSLog(
                @"bnb_mps_gemm_4bit dtype=%lld M=%lld N=%lld K=%lld bias=%d sched=%.3fms gpu=%.3fms done=%.3fms",
                (long long)dtype_flag, (long long)M, (long long)N, (long long)K, bias ? 1 : 0, sched_ms, gpu_ms, done_ms
            );
        }
    }
}

// Scalar parameter block for the fused 8-bit optimizer kernel. Must match the MSL
// `OptParams` struct field-for-field (all 4-byte members, no padding).
struct BnbOptParams {
    uint32_t n;
    uint32_t optimizer_id;
    float beta1;
    float one_minus_beta1;
    float beta2;
    float one_minus_beta2;
    float eps;
    float correction2;
    float update_scale;
    float wd_factor;
    float gnorm_scale;
};

// optimizer_update_8bit_blockwise (fused dequant-state -> optimizer update -> requant-state):
// g/p are the grad/param in the activation dtype (dtype_flag: 0 = fp32, 1 = fp16, 2 = bf16),
// state1/state2 are uint8 codes, qmap1/qmap2 the 256-entry fp32 dynamic maps, absmax1/absmax2
// the per-block fp32 maxima (read for dequant, overwritten with the post-update maxima).
// state2/qmap2/absmax2 may be NULL for 1-state optimizers (lion); the state1-side buffers are
// bound as placeholders and never dereferenced. optimizer_id: 3 = adam, 4 = lion. The float
// scalars are precomputed on the Python side in double precision (bias corrections, 1-beta,
// wd factor) so the kernel sees exactly the fp32 values torch's own kernels see.
// One threadgroup of 256 threads per state block; ONE command buffer, blocking wait.
extern "C" void bnb_mps_optimizer_update_8bit_blockwise(
    void* g, void* p, void* state1, void* state2, void* qmap1, void* qmap2, void* absmax1, void* absmax2, int64_t n,
    int64_t optimizer_id, int64_t dtype_flag, float beta1, float one_minus_beta1, float beta2, float one_minus_beta2,
    float eps, float correction2, float update_scale, float wd_factor, float gnorm_scale
) {
    @autoreleasepool {
        NSString* name = @"optimizer_update_8bit_blockwise_fp32";
        if (dtype_flag == 1) {
            name = @"optimizer_update_8bit_blockwise_fp16";
        } else if (dtype_flag == 2) {
            name = @"optimizer_update_8bit_blockwise_bf16";
        }
        id<MTLComputePipelineState> pso = get_pipeline(name);
        if (pso.maxTotalThreadsPerThreadgroup < 256) {
            NSLog(
                @"bitsandbytes: %@ supports only %lu threads/threadgroup (need 256)", name,
                (unsigned long)pso.maxTotalThreadsPerThreadgroup
            );
            abort();
        }

        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:(__bridge id<MTLBuffer>)g offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)p offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)state1 offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)(state2 ? state2 : state1) offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)qmap1 offset:0 atIndex:4];
        [enc setBuffer:(__bridge id<MTLBuffer>)(qmap2 ? qmap2 : qmap1) offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)absmax1 offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)(absmax2 ? absmax2 : absmax1) offset:0 atIndex:7];

        BnbOptParams prm;
        prm.n = (uint32_t)n;
        prm.optimizer_id = (uint32_t)optimizer_id;
        prm.beta1 = beta1;
        prm.one_minus_beta1 = one_minus_beta1;
        prm.beta2 = beta2;
        prm.one_minus_beta2 = one_minus_beta2;
        prm.eps = eps;
        prm.correction2 = correction2;
        prm.update_scale = update_scale;
        prm.wd_factor = wd_factor;
        prm.gnorm_scale = gnorm_scale;
        [enc setBytes:&prm length:sizeof(prm) atIndex:8];

        const int64_t blocks = (n + 255) / 256;
        [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// quantize_4bit: bounds (float32[15]), order (uint8[16]), A (float32[n]) ->
// out (uint8 packed, ceil(n/2)), absmax (float32[blocks]).
extern "C" void
    bnb_mps_quantize_4bit(void* bounds, void* order, void* A, void* out, void* absmax, int64_t n, int64_t blocksize) {
    @autoreleasepool {
        id<MTLComputePipelineState> pso = get_pipeline(@"quantize_4bit");
        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:(__bridge id<MTLBuffer>)bounds offset:0 atIndex:0];
        [enc setBuffer:(__bridge id<MTLBuffer>)order offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)A offset:0 atIndex:2];
        [enc setBuffer:(__bridge id<MTLBuffer>)out offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)absmax offset:0 atIndex:4];

        uint32_t n32 = (uint32_t)n;
        uint32_t bs32 = (uint32_t)blocksize;
        [enc setBytes:&n32 length:sizeof(n32) atIndex:5];
        [enc setBytes:&bs32 length:sizeof(bs32) atIndex:6];

        dispatch_per_block(enc, pso, (n + blocksize - 1) / blocksize);
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}
