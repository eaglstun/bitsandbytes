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

// quantize_blockwise: code (float32[256]), A (float32[n]) -> out (uint8[n]), absmax
// (float32[ceil(n/blocksize)]). All pointers are torch MPS tensor data_ptr() values,
// i.e. id<MTLBuffer> objects for offset-0 contiguous tensors.
extern "C" void bnb_mps_quantize_blockwise(void* code, void* A, void* out, void* absmax, int64_t n, int64_t blocksize) {
    @autoreleasepool {
        id<MTLBuffer> codeBuf = (__bridge id<MTLBuffer>)code;
        id<MTLBuffer> aBuf = (__bridge id<MTLBuffer>)A;
        id<MTLBuffer> outBuf = (__bridge id<MTLBuffer>)out;
        id<MTLBuffer> absmaxBuf = (__bridge id<MTLBuffer>)absmax;

        id<MTLComputePipelineState> pso = get_pipeline(@"quantize_blockwise");
        id<MTLCommandBuffer> cb = [get_queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];

        [enc setBuffer:codeBuf offset:0 atIndex:0];
        [enc setBuffer:aBuf offset:0 atIndex:1];
        [enc setBuffer:outBuf offset:0 atIndex:2];
        [enc setBuffer:absmaxBuf offset:0 atIndex:3];

        uint32_t n32 = (uint32_t)n;
        uint32_t bs32 = (uint32_t)blocksize;
        [enc setBytes:&n32 length:sizeof(n32) atIndex:4];
        [enc setBytes:&bs32 length:sizeof(bs32) atIndex:5];

        const NSUInteger num_blocks = (NSUInteger)((n + blocksize - 1) / blocksize);
        NSUInteger tg = pso.maxTotalThreadsPerThreadgroup;
        if (tg > 256) {
            tg = 256;
        }
        if (tg > num_blocks) {
            tg = num_blocks;
        }
        if (tg == 0) {
            tg = 1;
        }

        // One thread per block; non-uniform threadgroups (Apple GPUs support dispatchThreads).
        [enc dispatchThreads:MTLSizeMake(num_blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}
