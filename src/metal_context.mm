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
@property (nonatomic, strong) id<MTLComputePipelineState> contrastPSO;
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
        g_ctx.layer.allowsNextDrawableTimeout = NO;
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

        // CVMetalTextureCache for zero-copy CVPixelBuffer import
        [g_ctx createTexCacheWithDevice:g_ctx.device];

        // Compile contrast/brightness compute kernel
        NSString *src = @
            "using namespace metal;\n"
            "kernel void contrast_brightness(\n"
            "    texture2d<half, access::read_write> tex [[texture(0)]],\n"
            "    constant float &contrast   [[buffer(0)]],\n"
            "    constant float &brightness [[buffer(1)]],\n"
            "    constant float &pivot      [[buffer(2)]],\n"
            "    uint2 gid [[thread_position_in_grid]])\n"
            "{\n"
            "    if (gid.x >= tex.get_width() || gid.y >= tex.get_height()) return;\n"
            "    half4 c = tex.read(gid);\n"
            "    half3 rgb = c.rgb;\n"
            "    half p = half(pivot);\n"
            "    half con = half(contrast);\n"
            "    half bri = half(brightness / 255.0);\n"
            "    rgb = clamp(con * (rgb - p) + p + bri, half3(0), half3(1));\n"
            "    tex.write(half4(rgb, c.a), gid);\n"
            "}\n";
        NSError *err = nil;
        id<MTLLibrary> lib = [g_ctx.device newLibraryWithSource:src options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "[Metal] contrast kernel compile: %s\n",
                    err.localizedDescription.UTF8String);
        } else {
            id<MTLFunction> fn = [lib newFunctionWithName:@"contrast_brightness"];
            g_ctx.contrastPSO = [g_ctx.device newComputePipelineStateWithFunction:fn error:&err];
            if (!g_ctx.contrastPSO)
                fprintf(stderr, "[Metal] contrast PSO: %s\n",
                        err.localizedDescription.UTF8String);
        }
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
                    texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                width:widths[j]
                                               height:heights[j]
                                            mipmapped:NO];
            // ShaderWrite needed as blit destination; ShaderRead for ImGui sampling.
            // MTLStorageModeShared: CPU+GPU share unified memory on Apple Silicon
            // (replaceRegion: works without staging buffer for the CPU upload path)
            desc.usage       = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
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

// Import a CVPixelBuffer (BGRA, IOSurface-backed) as a Metal texture via
// CVMetalTextureCache and blit it into the per-camera output texture.
// VideoToolbox applies the correct color matrix (BT.601/BT.709, full/video
// range) internally based on the stream's metadata, so no shader is needed.
void metal_upload_pixelbuf(int cam_idx, CVPixelBufferRef pb, uint32_t w, uint32_t h) {
    if (![g_ctx texCache] || !g_ctx.cmd)
        return;

    @autoreleasepool {
        CVMetalTextureRef bgra_ref = NULL;
        CVReturn r = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault, [g_ctx texCache], pb, NULL,
            MTLPixelFormatBGRA8Unorm, w, h, 0, &bgra_ref);
        if (r != kCVReturnSuccess || !bgra_ref) {
            fprintf(stderr, "[Metal] CVMetalTextureCache BGRA failed: %d\n", r);
            return;
        }

        id<MTLTexture> src = CVMetalTextureGetTexture(bgra_ref);
        id<MTLTexture> dst = g_ctx.textures[cam_idx];

        // Blit from the IOSurface-backed VT texture into the stable output
        // texture that ImGui holds a reference to. Zero CPU involvement.
        id<MTLBlitCommandEncoder> blit = [g_ctx.cmd blitCommandEncoder];
        [blit copyFromTexture:src
                  sourceSlice:0 sourceLevel:0
                 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(w, h, 1)
                    toTexture:dst
             destinationSlice:0 destinationLevel:0
            destinationOrigin:MTLOriginMake(0, 0, 0)];
        [blit endEncoding];

        CFRelease(bgra_ref);
    }
}

ImTextureID metal_get_texture_id(int cam_idx) {
    // ImTextureID is ImU64; store ObjC pointer as integer via bridge+intptr_t cast
    return (ImTextureID)(intptr_t)(__bridge void*)g_ctx.textures[cam_idx];
}

void metal_apply_contrast_brightness(int cam_idx, float contrast, float brightness, bool pivot_midgray) {
    if (!g_ctx.contrastPSO || !g_ctx.cmd) return;
    if (contrast == 1.0f && brightness == 0.0f) return;

    @autoreleasepool {
        id<MTLTexture> tex = g_ctx.textures[cam_idx];
        float pivot = pivot_midgray ? 0.5f : 0.0f;

        id<MTLComputeCommandEncoder> enc = [g_ctx.cmd computeCommandEncoder];
        [enc setComputePipelineState:g_ctx.contrastPSO];
        [enc setTexture:tex atIndex:0];
        [enc setBytes:&contrast length:sizeof(float) atIndex:0];
        [enc setBytes:&brightness length:sizeof(float) atIndex:1];
        [enc setBytes:&pivot length:sizeof(float) atIndex:2];

        MTLSize grid = MTLSizeMake(tex.width, tex.height, 1);
        MTLSize tg   = MTLSizeMake(16, 16, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
    }
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
