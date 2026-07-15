#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <dlfcn.h>
#include <libgen.h>
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
        [cb commit];
        [cb waitUntilCompleted];

        // Kernel-only timing probe (BNB_MPS_PROFILE=1): wall-clock around this call includes
        // the cross-queue sync + encode + blocking wait; GPUStartTime/GPUEndTime isolates the
        // kernel itself for bandwidth accounting.
        static const bool profile = getenv("BNB_MPS_PROFILE") != nullptr;
        if (profile) {
            const double gpu_ms = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
            NSLog(@"bnb_mps_gemv_4bit %@ N=%lld K=%lld gpu=%.3fms", name, (long long)N, (long long)K, gpu_ms);
        }
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
