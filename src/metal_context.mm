#ifdef __APPLE__

#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#import <CoreVideo/CoreVideo.h>
#import <CoreVideo/CVMetalTextureCache.h>

// Expose glfwGetCocoaWindow
#define GLFW_EXPOSE_NATIVE_COCOA
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "imgui.h"
#include "imgui_impl_metal.h"
#include "metal_context.h"

#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Internal Objective-C class — owns all Metal objects via ARC
// ---------------------------------------------------------------------------

@interface MetalCtx : NSObject
@property (nonatomic, strong) id<MTLDevice>             device;
@property (nonatomic, strong) id<MTLCommandQueue>       queue;
@property (nonatomic, strong) CAMetalLayer             *layer;
@property (nonatomic, strong) NSMutableArray           *textures;    // id<MTLTexture>
@property (nonatomic, strong) id<CAMetalDrawable>       drawable;
@property (nonatomic, strong) MTLRenderPassDescriptor  *rpd;
@property (nonatomic, strong) id<MTLCommandBuffer>      cmd;
@property (nonatomic, strong) id<MTLComputePipelineState> nv12PSO;
@end

@implementation MetalCtx {
    CVMetalTextureCacheRef _texCache;
}

- (CVMetalTextureCacheRef)texCache { return _texCache; }

- (void)createTexCacheWithDevice:(id<MTLDevice>)dev {
    CVReturn r = CVMetalTextureCacheCreate(kCFAllocatorDefault, NULL, dev, NULL, &_texCache);
    if (r != kCVReturnSuccess)
        fprintf(stderr, "[Metal] CVMetalTextureCacheCreate failed: %d\n", r);
}

- (void)dealloc {
    if (_texCache) {
        CVMetalTextureCacheFlush(_texCache, 0);
        CFRelease(_texCache);
        _texCache = NULL;
    }
}
@end

static MetalCtx *g_ctx = nil;
static uint64_t  g_frame_count = 0;

// ---------------------------------------------------------------------------
// NV12→RGBA compute shader (BT.601 full-range)
// ---------------------------------------------------------------------------

static NSString * const kNV12Shader = @""
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "kernel void nv12_to_rgba(\n"
    "    texture2d<float, access::read>  y_plane  [[texture(0)]],\n"
    "    texture2d<float, access::read>  uv_plane [[texture(1)]],\n"
    "    texture2d<float, access::write> out_rgba [[texture(2)]],\n"
    "    uint2 gid [[thread_position_in_grid]])\n"
    "{\n"
    // No bounds check needed: dispatchThreads sizes the grid exactly to
    // (w, h) so Metal guarantees gid is always within texture bounds.
    "    float  y  = y_plane.read(gid).r;\n"
    "    float2 uv = uv_plane.read(gid / 2).rg - 0.5f;\n"
    "    float r = clamp(y + 1.402f   * uv.y,                   0.0f, 1.0f);\n"
    "    float g = clamp(y - 0.34414f * uv.x - 0.71414f * uv.y, 0.0f, 1.0f);\n"
    "    float b = clamp(y + 1.772f   * uv.x,                   0.0f, 1.0f);\n"
    "    out_rgba.write(float4(r, g, b, 1.0f), gid);\n"
    "}\n";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void metal_init(GLFWwindow *window) {
    @autoreleasepool {
        g_ctx = [[MetalCtx alloc] init];

        g_ctx.device = MTLCreateSystemDefaultDevice();
        if (!g_ctx.device) {
            fprintf(stderr, "[Metal] No Metal device found\n");
            exit(EXIT_FAILURE);
        }

        g_ctx.queue    = [g_ctx.device newCommandQueue];
        g_ctx.textures = [NSMutableArray array];

        // Attach CAMetalLayer to the GLFW Cocoa window
        NSWindow *nswin = glfwGetCocoaWindow(window);
        g_ctx.layer = [CAMetalLayer layer];
        g_ctx.layer.device          = g_ctx.device;
        g_ctx.layer.pixelFormat     = MTLPixelFormatBGRA8Unorm;
        g_ctx.layer.framebufferOnly = YES;
        nswin.contentView.layer      = g_ctx.layer;
        nswin.contentView.wantsLayer = YES;

        // Match contentsScale to the screen's backing scale factor so the
        // drawable (and ImGui's viewport) are at the correct physical resolution.
        // Without this, CAMetalLayer defaults to 1x while GLFW reports a 2x
        // framebuffer, causing all UI elements to render at double logical size.
        CGFloat scale = nswin.screen ? nswin.screen.backingScaleFactor : 1.0;
        g_ctx.layer.contentsScale = scale;
        CGSize drawableSize = [nswin.contentView convertSizeToBacking:nswin.contentView.bounds.size];
        g_ctx.layer.drawableSize = drawableSize;

        // Build NV12→RGBA compute pipeline
        NSError *err = nil;
        id<MTLLibrary> lib = [g_ctx.device newLibraryWithSource:kNV12Shader
                                                        options:nil
                                                          error:&err];
        if (!lib) {
            fprintf(stderr, "[Metal] NV12 shader compile error: %s\n",
                    err.localizedDescription.UTF8String);
        } else {
            id<MTLFunction> fn = [lib newFunctionWithName:@"nv12_to_rgba"];
            g_ctx.nv12PSO = [g_ctx.device newComputePipelineStateWithFunction:fn
                                                                        error:&err];
            if (!g_ctx.nv12PSO)
                fprintf(stderr, "[Metal] NV12 pipeline error: %s\n",
                        err.localizedDescription.UTF8String);
        }

        // CVMetalTextureCache for zero-copy CVPixelBuffer import
        [g_ctx createTexCacheWithDevice:g_ctx.device];
    }
}

void metal_init_imgui() {
    ImGui_ImplMetal_Init(g_ctx.device);
}

void metal_allocate_textures(int num_cams, uint32_t *widths, uint32_t *heights) {
    @autoreleasepool {
        [g_ctx.textures removeAllObjects];
        for (int j = 0; j < num_cams; j++) {
            MTLTextureDescriptor *desc =
                [MTLTextureDescriptor
                    texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                width:widths[j]
                                               height:heights[j]
                                            mipmapped:NO];
            // ShaderWrite needed for NV12 compute output
            desc.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
            // MTLStorageModeShared: CPU+GPU share unified memory on Apple Silicon
            // (replaceRegion: works without staging buffer)
            desc.storageMode = MTLStorageModeShared;
            id<MTLTexture> tex = [g_ctx.device newTextureWithDescriptor:desc];
            [g_ctx.textures addObject:tex];
        }
    }
}

bool metal_begin_frame() {
    // Flush stale CVMetalTextureCache entries every 60 frames to prevent
    // recycled CVPixelBuffers from accumulating unreleased cache wrappers.
    if (++g_frame_count % 60 == 0)
        CVMetalTextureCacheFlush([g_ctx texCache], 0);

    g_ctx.drawable = [g_ctx.layer nextDrawable];
    if (!g_ctx.drawable)
        return false;

    g_ctx.rpd = [MTLRenderPassDescriptor renderPassDescriptor];
    g_ctx.rpd.colorAttachments[0].texture     = g_ctx.drawable.texture;
    g_ctx.rpd.colorAttachments[0].loadAction  = MTLLoadActionClear;
    g_ctx.rpd.colorAttachments[0].clearColor  = MTLClearColorMake(0.1, 0.1, 0.1, 1.0);
    g_ctx.rpd.colorAttachments[0].storeAction = MTLStoreActionStore;

    g_ctx.cmd = [g_ctx.queue commandBuffer];

    ImGui_ImplMetal_NewFrame(g_ctx.rpd);
    return true;
}

// Phase 1: Upload pre-converted RGBA bytes directly to MTLTexture.
// On Apple Silicon, MTLStorageModeShared textures are CPU-writable
// with no staging buffer or copy needed.
void metal_upload_texture(int cam_idx, const uint8_t *rgba, uint32_t w, uint32_t h) {
    id<MTLTexture> tex = g_ctx.textures[cam_idx];
    [tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
           mipmapLevel:0
             withBytes:rgba
           bytesPerRow:w * 4];
}

// Phase 2: Import a CVPixelBuffer (NV12) as MTLTextures via CVMetalTextureCache
// and run a compute shader to convert NV12→RGBA in the output texture.
// The CVPixelBuffer is IOSurface-backed, so this is zero-copy on the CPU side.
void metal_upload_pixelbuf(int cam_idx, CVPixelBufferRef pb, uint32_t w, uint32_t h) {
    if (!g_ctx.nv12PSO || ![g_ctx texCache] || !g_ctx.cmd)
        return;

    @autoreleasepool {
        CVMetalTextureCacheRef cache = [g_ctx texCache];

        CVMetalTextureRef y_ref  = NULL;
        CVMetalTextureRef uv_ref = NULL;

        CVReturn r;
        r = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pb, NULL,
            MTLPixelFormatR8Unorm, w, h, 0, &y_ref);
        if (r != kCVReturnSuccess || !y_ref) {
            fprintf(stderr, "[Metal] CVMetalTextureCache Y failed: %d\n", r);
            return;
        }

        r = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, cache, pb, NULL,
            MTLPixelFormatRG8Unorm, w / 2, h / 2, 1, &uv_ref);
        if (r != kCVReturnSuccess || !uv_ref) {
            fprintf(stderr, "[Metal] CVMetalTextureCache UV failed: %d\n", r);
            CFRelease(y_ref);
            return;
        }

        id<MTLTexture> y_tex  = CVMetalTextureGetTexture(y_ref);
        id<MTLTexture> uv_tex = CVMetalTextureGetTexture(uv_ref);
        id<MTLTexture> out    = g_ctx.textures[cam_idx];

        // Dispatch compute on the frame's command buffer
        id<MTLComputeCommandEncoder> enc = [g_ctx.cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx.nv12PSO];
        [enc setTexture:y_tex  atIndex:0];
        [enc setTexture:uv_tex atIndex:1];
        [enc setTexture:out    atIndex:2];
        // dispatchThreads sizes the grid exactly to the texture dimensions;
        // Metal handles non-multiple-of-threadgroup remainders automatically.
        MTLSize tg     = MTLSizeMake(16, 16, 1);
        MTLSize pixels = MTLSizeMake(w, h, 1);
        [enc dispatchThreads:pixels threadsPerThreadgroup:tg];
        [enc endEncoding];

        CFRelease(y_ref);
        CFRelease(uv_ref);
    }
}

ImTextureID metal_get_texture_id(int cam_idx) {
    // ImTextureID is ImU64; store ObjC pointer as integer via bridge+intptr_t cast
    return (ImTextureID)(intptr_t)(__bridge void*)g_ctx.textures[cam_idx];
}

void metal_end_frame() {
    @autoreleasepool {
        id<MTLRenderCommandEncoder> enc =
            [g_ctx.cmd renderCommandEncoderWithDescriptor:g_ctx.rpd];
        ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), g_ctx.cmd, enc);
        [enc endEncoding];

        [g_ctx.cmd presentDrawable:g_ctx.drawable];
        [g_ctx.cmd commit];

        g_ctx.cmd      = nil;
        g_ctx.drawable = nil;
        g_ctx.rpd      = nil;
    }
}

void metal_cleanup() {
    @autoreleasepool {
        // Drain GPU before releasing resources
        id<MTLCommandBuffer> flush = [g_ctx.queue commandBuffer];
        [flush commit];
        [flush waitUntilCompleted];

        ImGui_ImplMetal_Shutdown();
        [g_ctx.textures removeAllObjects];
        g_ctx = nil;
    }
}

#endif // __APPLE__
