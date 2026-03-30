#ifdef __APPLE__
#ifdef RED_HAS_MUJOCO

#import <Metal/Metal.h>
#import <simd/simd.h>
#include "mujoco_metal_renderer.h"
#include "mujoco_context.h"
#include <mujoco.h>
#include <vector>
#include <cmath>

// ---------------------------------------------------------------------------
// Shader source — renders capsules/spheres/boxes as lit 3D primitives
// ---------------------------------------------------------------------------

static const char *kShaderSource = R"(
#include <metal_stdlib>
using namespace metal;

struct VertexIn {
    float3 position [[attribute(0)]];
    float3 normal   [[attribute(1)]];
};

struct Uniforms {
    float4x4 mvp;
    float4x4 model_mat;
    float4   normal_col0;
    float4   normal_col1;
    float4   normal_col2;
    float4   color;
};

struct SceneUniforms {
    float4x4 view_proj;
    float3   light_dir;
    float    _pad0;
    float3   light_dir2;
    float    _pad1;
    float3   eye_pos;
    float    _pad2;
};

struct VertexOut {
    float4 position [[position]];
    float3 normal;
    float3 world_pos;
    float4 color;
};

vertex VertexOut vertex_main(VertexIn in [[stage_in]],
                             constant Uniforms &u [[buffer(1)]],
                             constant SceneUniforms &su [[buffer(2)]]) {
    VertexOut out;
    float4 wp = u.model_mat * float4(in.position, 1.0);
    out.world_pos = wp.xyz;
    out.position = su.view_proj * wp;
    float3x3 normal_mat = float3x3(u.normal_col0.xyz, u.normal_col1.xyz, u.normal_col2.xyz);
    out.normal = normalize(normal_mat * in.normal);
    out.color = u.color;
    return out;
}

fragment float4 fragment_main(VertexOut in [[stage_in]],
                              constant SceneUniforms &su [[buffer(2)]],
                              bool front_facing [[front_facing]]) {
    float3 N = normalize(in.normal);
    if (!front_facing) N = -N; // two-sided lighting for mesh geoms
    float3 V = normalize(su.eye_pos - in.world_pos);

    // Two-light setup: key light (above) + fill light (side)
    float3 L1 = normalize(su.light_dir);
    float3 L2 = normalize(su.light_dir2);
    float3 H1 = normalize(L1 + V);
    float3 H2 = normalize(L2 + V);

    float ambient = 0.20;
    float diffuse = max(dot(N, L1), 0.0) * 0.50
                  + max(dot(N, L2), 0.0) * 0.25;
    float spec = pow(max(dot(N, H1), 0.0), 32.0) * 0.25
               + pow(max(dot(N, H2), 0.0), 32.0) * 0.10;

    float3 color = in.color.rgb * (ambient + diffuse) + float3(spec);
    return float4(color, in.color.a);
}

// --- Floor shader: checkerboard tiles with grid lines and reflections ---

struct FloorUniforms {
    float4x4 mvp;
    float    half_size;    // arena half-extent in meters
    int      grid_divs;    // number of grid divisions per axis
    float    _pad[2];
};

fragment float4 fragment_floor(VertexOut in [[stage_in]],
                               constant SceneUniforms &su [[buffer(2)]],
                               constant FloorUniforms &fu [[buffer(3)]]) {
    float3 N = float3(0, 0, 1);
    float3 V = normalize(su.eye_pos - in.world_pos);

    // Tile coordinates
    float tile_size = (2.0 * fu.half_size) / float(fu.grid_divs);
    float2 uv = (in.world_pos.xy + float2(fu.half_size)) / tile_size;
    int2 tile = int2(floor(uv));
    float2 frac_uv = fract(uv);

    // Checkerboard: alternating dark/light blue-gray tiles
    bool checker = ((tile.x + tile.y) & 1) == 0;
    float3 tile_dark  = float3(0.22, 0.28, 0.36);
    float3 tile_light = float3(0.28, 0.35, 0.44);
    float3 base_color = checker ? tile_dark : tile_light;

    // Grid lines: bright lines at tile edges
    float line_w = 0.015; // fraction of tile
    float lx = min(frac_uv.x, 1.0 - frac_uv.x);
    float ly = min(frac_uv.y, 1.0 - frac_uv.y);
    float line_dist = min(lx, ly);
    float line = 1.0 - smoothstep(0.0, line_w, line_dist);
    float3 line_color = float3(0.75, 0.80, 0.85);
    base_color = mix(base_color, line_color, line * 0.8);

    // Two-light diffuse + strong specular for reflective look
    float3 L1 = normalize(su.light_dir);
    float3 L2 = normalize(su.light_dir2);
    float3 H1 = normalize(L1 + V);
    float3 H2 = normalize(L2 + V);

    float ambient = 0.25;
    float diffuse = max(dot(N, L1), 0.0) * 0.45
                  + max(dot(N, L2), 0.0) * 0.20;
    float spec = pow(max(dot(N, H1), 0.0), 64.0) * 0.45
               + pow(max(dot(N, H2), 0.0), 64.0) * 0.15;

    float3 color = base_color * (ambient + diffuse) + float3(spec);
    return float4(color, in.color.a);
}

// --- Fullscreen blit: draws a texture as background ---

struct BlitParams {
    float2 uv_scale;   // 1/zoom
    float2 uv_offset;  // pan offset
};

struct BlitVertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex BlitVertexOut blit_vertex(uint vid [[vertex_id]],
                                 constant BlitParams &bp [[buffer(0)]]) {
    BlitVertexOut out;
    float2 base_uv = float2((vid << 1) & 2, vid & 2);
    out.position = float4(base_uv * 2.0 - 1.0, 0.0, 1.0);
    // Apply zoom/pan to UV: zoom around center, then offset
    float2 uv = base_uv;
    uv.y = 1.0 - uv.y; // flip Y
    uv = (uv - 0.5) * bp.uv_scale + 0.5 + bp.uv_offset;
    out.uv = uv;
    return out;
}

fragment float4 blit_fragment(BlitVertexOut in [[stage_in]],
                              texture2d<float> tex [[texture(0)]]) {
    if (in.uv.x < 0.0 || in.uv.x > 1.0 || in.uv.y < 0.0 || in.uv.y > 1.0)
        return float4(0, 0, 0, 1);
    constexpr sampler s(filter::linear);
    return tex.sample(s, in.uv);
}
)";

// ---------------------------------------------------------------------------
// Unit geometry generators
// ---------------------------------------------------------------------------

struct Vertex {
    simd_float3 position;
    simd_float3 normal;
};

static std::vector<Vertex> make_unit_sphere(int rings, int sectors) {
    std::vector<Vertex> verts;
    for (int r = 0; r <= rings; r++) {
        float phi = M_PI * r / rings;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * M_PI * s / sectors;
            float sp = sinf(phi), cp = cosf(phi);
            float st = sinf(theta), ct = cosf(theta);
            // Z-up: pole at (0,0,1). Matches capsule winding for consistent
            // CCW front-face order with MTLWindingCounterClockwise.
            simd_float3 n = {sp * ct, sp * st, cp};
            verts.push_back({n, n});
        }
    }
    return verts;
}

static std::vector<uint32_t> make_sphere_indices(int rings, int sectors) {
    std::vector<uint32_t> idx;
    for (int r = 0; r < rings; r++) {
        for (int s = 0; s < sectors; s++) {
            uint32_t a = r * (sectors + 1) + s;
            uint32_t b = a + sectors + 1;
            idx.insert(idx.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }
    return idx;
}

// Unit capsule: cylinder from z=-0.5 to z=+0.5, radius 1, with hemisphere caps.
// MuJoCo capsules extend along the Z-axis of the geom frame.
static std::vector<Vertex> make_unit_capsule(int rings, int sectors) {
    std::vector<Vertex> verts;
    // Top hemisphere (z = +0.5 pole)
    for (int r = 0; r <= rings / 2; r++) {
        float phi = M_PI * r / rings;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * M_PI * s / sectors;
            float sp = sinf(phi), cp = cosf(phi);
            float st = sinf(theta), ct = cosf(theta);
            // Standard sphere with pole at +Z, rotated so Z is up
            simd_float3 n = {sp * ct, sp * st, cp};
            simd_float3 p = {n.x, n.y, n.z + 0.5f};
            verts.push_back({p, n});
        }
    }
    // Cylinder body (top and bottom rings)
    for (int cap = 0; cap < 2; cap++) {
        float z = cap ? -0.5f : 0.5f;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * M_PI * s / sectors;
            simd_float3 n = {cosf(theta), sinf(theta), 0.0f};
            simd_float3 p = {n.x, n.y, z};
            verts.push_back({p, n});
        }
    }
    // Bottom hemisphere (z = -0.5 pole)
    for (int r = rings / 2; r <= rings; r++) {
        float phi = M_PI * r / rings;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * M_PI * s / sectors;
            float sp = sinf(phi), cp = cosf(phi);
            float st = sinf(theta), ct = cosf(theta);
            simd_float3 n = {sp * ct, sp * st, cp};
            simd_float3 p = {n.x, n.y, n.z - 0.5f};
            verts.push_back({p, n});
        }
    }
    return verts;
}

static std::vector<uint32_t> make_capsule_indices(int rings, int sectors) {
    std::vector<uint32_t> idx;
    // Vertex rows: top hemi (rings/2+1) + cylinder (2) + bottom hemi (rings/2+1)
    int total_rows = rings + 4;
    int stride = sectors + 1;
    for (int r = 0; r < total_rows - 1; r++) {
        for (int s = 0; s < sectors; s++) {
            uint32_t a = r * stride + s;
            uint32_t b = a + stride;
            idx.insert(idx.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
    }
    return idx;
}

// Unit cylinder tube (open ends): radius 1, extends from z=-1 to z=+1.
// When scaled by half_len in Z, gives correct extent [-half_len, +half_len].
static std::vector<Vertex> make_unit_cylinder(int sectors) {
    std::vector<Vertex> verts;
    for (int cap = 0; cap < 2; cap++) {
        float z = cap ? -1.0f : 1.0f;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * M_PI * s / sectors;
            simd_float3 n = {cosf(theta), sinf(theta), 0.0f};
            simd_float3 p = {n.x, n.y, z};
            verts.push_back({p, n});
        }
    }
    return verts;
}

static std::vector<uint32_t> make_cylinder_indices(int sectors) {
    std::vector<uint32_t> idx;
    int stride = sectors + 1;
    for (int s = 0; s < sectors; s++) {
        uint32_t a = s, b = a + stride;
        idx.insert(idx.end(), {a, b, a + 1, a + 1, b, b + 1});
    }
    return idx;
}

// Unit box: [-1,1]^3 (scaled to half-extents by model matrix)
static simd_float3 sf3(float x, float y, float z) {
    return (simd_float3){x, y, z};
}

static std::vector<Vertex> make_unit_box() {
    std::vector<Vertex> v;
    auto face = [&](simd_float3 n, simd_float3 u, simd_float3 r) {
        v.push_back({n + u + r, n}); v.push_back({n - u + r, n});
        v.push_back({n - u - r, n}); v.push_back({n + u - r, n});
    };
    face(sf3(0,0,1), sf3(1,0,0), sf3(0,1,0));   // +Z
    face(sf3(0,0,-1), sf3(-1,0,0), sf3(0,1,0)); // -Z
    face(sf3(1,0,0), sf3(0,0,1), sf3(0,1,0));   // +X
    face(sf3(-1,0,0), sf3(0,0,-1), sf3(0,1,0)); // -X
    face(sf3(0,1,0), sf3(1,0,0), sf3(0,0,1));   // +Y
    face(sf3(0,-1,0), sf3(1,0,0), sf3(0,0,-1)); // -Y
    return v;
}

static std::vector<uint32_t> make_box_indices() {
    std::vector<uint32_t> idx;
    for (uint32_t f = 0; f < 6; f++) {
        uint32_t b = f * 4;
        idx.insert(idx.end(), {b, b+1, b+2, b, b+2, b+3});
    }
    return idx;
}

// ---------------------------------------------------------------------------
// Renderer state
// ---------------------------------------------------------------------------

struct MujocoRenderer {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLRenderPipelineState> pipeline;
    id<MTLRenderPipelineState> floor_pipeline;
    id<MTLRenderPipelineState> blit_pipeline;
    id<MTLDepthStencilState>   depth_state;
    id<MTLTexture>             color_tex;
    id<MTLTexture>             depth_tex;
    uint32_t width, height;

    // Unit geometry buffers
    id<MTLBuffer> sphere_vb, sphere_ib;
    int sphere_idx_count;
    id<MTLBuffer> capsule_vb, capsule_ib;
    int capsule_idx_count;
    id<MTLBuffer> box_vb, box_ib;
    int box_idx_count;
    id<MTLBuffer> cylinder_vb, cylinder_ib;
    int cylinder_idx_count;
    id<MTLBuffer> floor_vb, floor_ib;

    // Dynamic skin buffers (rebuilt each frame from mjvScene skin data)
    id<MTLBuffer> skin_vb;
    id<MTLBuffer> skin_ib;
    int skin_vert_count;
    int skin_face_count;
    bool skin_ib_built;  // face indices are static, only build once
};

static id<MTLBuffer> make_buffer(id<MTLDevice> dev, const void *data, size_t size) {
    return [dev newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
}

static void create_textures(MujocoRenderer *r) {
    MTLTextureDescriptor *cd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                width:r->width height:r->height mipmapped:NO];
    cd.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    cd.storageMode = MTLStorageModeShared;
    r->color_tex = [r->device newTextureWithDescriptor:cd];

    MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                width:r->width height:r->height mipmapped:NO];
    dd.usage = MTLTextureUsageRenderTarget;
    dd.storageMode = MTLStorageModePrivate;
    r->depth_tex = [r->device newTextureWithDescriptor:dd];
}

MujocoRenderer *mujoco_renderer_create(uint32_t width, uint32_t height) {
    MujocoRenderer *r = new MujocoRenderer();
    r->width = width;
    r->height = height;
    r->device = MTLCreateSystemDefaultDevice();
    r->queue = [r->device newCommandQueue];

    // Compile shaders
    NSError *err = nil;
    NSString *src = [NSString stringWithUTF8String:kShaderSource];
    id<MTLLibrary> lib = [r->device newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        fprintf(stderr, "[MuJoCo Metal] Shader compile failed: %s\n",
                [[err localizedDescription] UTF8String]);
        delete r;
        return nullptr;
    }

    // Pipeline
    MTLRenderPipelineDescriptor *pd = [[MTLRenderPipelineDescriptor alloc] init];
    pd.vertexFunction = [lib newFunctionWithName:@"vertex_main"];
    pd.fragmentFunction = [lib newFunctionWithName:@"fragment_main"];
    pd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
    pd.colorAttachments[0].blendingEnabled = YES;
    pd.colorAttachments[0].sourceRGBBlendFactor = MTLBlendFactorSourceAlpha;
    pd.colorAttachments[0].destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pd.colorAttachments[0].sourceAlphaBlendFactor = MTLBlendFactorOne;
    pd.colorAttachments[0].destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    pd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;

    // Vertex descriptor
    MTLVertexDescriptor *vd = [[MTLVertexDescriptor alloc] init];
    vd.attributes[0].format = MTLVertexFormatFloat3;
    vd.attributes[0].offset = 0;
    vd.attributes[0].bufferIndex = 0;
    vd.attributes[1].format = MTLVertexFormatFloat3;
    vd.attributes[1].offset = sizeof(simd_float3);
    vd.attributes[1].bufferIndex = 0;
    vd.layouts[0].stride = sizeof(Vertex);
    pd.vertexDescriptor = vd;

    r->pipeline = [r->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!r->pipeline) {
        fprintf(stderr, "[MuJoCo Metal] Pipeline creation failed: %s\n",
                [[err localizedDescription] UTF8String]);
        delete r;
        return nullptr;
    }

    // Floor pipeline (same vertex shader, different fragment shader)
    pd.fragmentFunction = [lib newFunctionWithName:@"fragment_floor"];
    r->floor_pipeline = [r->device newRenderPipelineStateWithDescriptor:pd error:&err];
    if (!r->floor_pipeline) {
        fprintf(stderr, "[MuJoCo Metal] Floor pipeline creation failed: %s\n",
                [[err localizedDescription] UTF8String]);
        // Non-fatal: fall back to regular pipeline for floor
        r->floor_pipeline = r->pipeline;
    }

    // Blit pipeline (fullscreen texture draw, no vertex input, no depth)
    {
        MTLRenderPipelineDescriptor *bpd = [[MTLRenderPipelineDescriptor alloc] init];
        bpd.vertexFunction = [lib newFunctionWithName:@"blit_vertex"];
        bpd.fragmentFunction = [lib newFunctionWithName:@"blit_fragment"];
        bpd.colorAttachments[0].pixelFormat = MTLPixelFormatBGRA8Unorm;
        bpd.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
        r->blit_pipeline = [r->device newRenderPipelineStateWithDescriptor:bpd error:&err];
        if (!r->blit_pipeline)
            fprintf(stderr, "[MuJoCo Metal] Blit pipeline failed: %s\n",
                    [[err localizedDescription] UTF8String]);
    }

    // Depth stencil
    MTLDepthStencilDescriptor *dsd = [[MTLDepthStencilDescriptor alloc] init];
    dsd.depthCompareFunction = MTLCompareFunctionLess;
    dsd.depthWriteEnabled = YES;
    r->depth_state = [r->device newDepthStencilStateWithDescriptor:dsd];

    // Create offscreen textures
    create_textures(r);

    // Generate unit geometry
    int rings = 12, sectors = 16;

    auto sv = make_unit_sphere(rings, sectors);
    auto si = make_sphere_indices(rings, sectors);
    r->sphere_vb = make_buffer(r->device, sv.data(), sv.size() * sizeof(Vertex));
    r->sphere_ib = make_buffer(r->device, si.data(), si.size() * sizeof(uint32_t));
    r->sphere_idx_count = (int)si.size();

    auto cv = make_unit_capsule(rings, sectors);
    auto ci = make_capsule_indices(rings, sectors);
    r->capsule_vb = make_buffer(r->device, cv.data(), cv.size() * sizeof(Vertex));
    r->capsule_ib = make_buffer(r->device, ci.data(), ci.size() * sizeof(uint32_t));
    r->capsule_idx_count = (int)ci.size();

    auto bv = make_unit_box();
    auto bi = make_box_indices();
    r->box_vb = make_buffer(r->device, bv.data(), bv.size() * sizeof(Vertex));
    r->box_ib = make_buffer(r->device, bi.data(), bi.size() * sizeof(uint32_t));
    r->box_idx_count = (int)bi.size();

    auto cyv = make_unit_cylinder(sectors);
    auto cyi = make_cylinder_indices(sectors);
    r->cylinder_vb = make_buffer(r->device, cyv.data(), cyv.size() * sizeof(Vertex));
    r->cylinder_ib = make_buffer(r->device, cyi.data(), cyi.size() * sizeof(uint32_t));
    r->cylinder_idx_count = (int)cyi.size();

    // Pre-allocate floor quad (arena: 1828mm x 1828mm centered at origin)
    float half = 0.914f;
    Vertex floor_verts[4] = {
        {{-half, -half, 0}, {0, 0, 1}},
        {{ half, -half, 0}, {0, 0, 1}},
        {{ half,  half, 0}, {0, 0, 1}},
        {{-half,  half, 0}, {0, 0, 1}},
    };
    uint32_t floor_idx[6] = {0, 1, 2, 0, 2, 3};
    r->floor_vb = make_buffer(r->device, floor_verts, sizeof(floor_verts));
    r->floor_ib = make_buffer(r->device, floor_idx, sizeof(floor_idx));

    // Skin buffers initialized lazily on first render
    r->skin_vb = nil;
    r->skin_ib = nil;
    r->skin_vert_count = 0;
    r->skin_face_count = 0;
    r->skin_ib_built = false;

    return r;
}

void mujoco_renderer_destroy(MujocoRenderer *r) {
    if (!r) return;
    @autoreleasepool {
        // Wait for any in-flight GPU work to complete before releasing resources
        if (r->queue) {
            id<MTLCommandBuffer> fence = [r->queue commandBuffer];
            [fence commit];
            [fence waitUntilCompleted];
        }
        // Zero all Obj-C references — they'll be released by ARC
        r->color_tex = nil;
        r->depth_tex = nil;
        r->sphere_vb = nil;  r->sphere_ib = nil;
        r->capsule_vb = nil; r->capsule_ib = nil;
        r->box_vb = nil;     r->box_ib = nil;
        r->cylinder_vb = nil; r->cylinder_ib = nil;
        r->floor_vb = nil;    r->floor_ib = nil;
        r->skin_vb = nil;    r->skin_ib = nil;
        r->pipeline = nil;
        r->floor_pipeline = nil;
        r->blit_pipeline = nil;
        r->depth_state = nil;
        r->queue = nil;
        r->device = nil;
    }
    delete r;
}

void mujoco_renderer_resize(MujocoRenderer *r, uint32_t width, uint32_t height) {
    if (!r || (r->width == width && r->height == height)) return;
    r->width = width;
    r->height = height;
    r->color_tex = nil; // release old textures before creating new
    r->depth_tex = nil;
    create_textures(r);
}

// Build a 4x4 model matrix from mjvGeom pos[3] + mat[9] + size[3].
// MuJoCo g.mat is 3x3 row-major: mat[0..2] = row 0, mat[3..5] = row 1, mat[6..8] = row 2.
// Metal wants column-major, so we transpose: column j = (mat[j], mat[3+j], mat[6+j]).
static simd_float4x4 geom_model_matrix(const mjvGeom &g, float sx, float sy, float sz) {
    simd_float4x4 m;
    m.columns[0] = {(float)g.mat[0]*sx, (float)g.mat[3]*sx, (float)g.mat[6]*sx, 0};
    m.columns[1] = {(float)g.mat[1]*sy, (float)g.mat[4]*sy, (float)g.mat[7]*sy, 0};
    m.columns[2] = {(float)g.mat[2]*sz, (float)g.mat[5]*sz, (float)g.mat[8]*sz, 0};
    m.columns[3] = {(float)g.pos[0], (float)g.pos[1], (float)g.pos[2], 1.0f};
    return m;
}

// Extract normal matrix columns as float4 (w=0) for uniform buffer alignment.
// For non-uniform scale, this is approximate but acceptable for visualization.
static void extract_normal_columns(simd_float4x4 m,
                                   simd_float4 &col0, simd_float4 &col1, simd_float4 &col2) {
    col0 = {m.columns[0].x, m.columns[0].y, m.columns[0].z, 0};
    col1 = {m.columns[1].x, m.columns[1].y, m.columns[1].z, 0};
    col2 = {m.columns[2].x, m.columns[2].y, m.columns[2].z, 0};
}

static simd_float4x4 look_at(simd_float3 eye, simd_float3 center, simd_float3 up) {
    simd_float3 f = simd_normalize(center - eye);
    simd_float3 s = simd_normalize(simd_cross(f, up));
    simd_float3 u = simd_cross(s, f);
    simd_float4x4 m;
    m.columns[0] = {s.x, u.x, -f.x, 0};
    m.columns[1] = {s.y, u.y, -f.y, 0};
    m.columns[2] = {s.z, u.z, -f.z, 0};
    m.columns[3] = {-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1};
    return m;
}

// Metal NDC: Z maps to [0, 1] (not [-1, 1] like OpenGL).
// This gives full depth buffer precision.
static simd_float4x4 perspective(float fov_y, float aspect, float near, float far) {
    float f = 1.0f / tanf(fov_y * 0.5f);
    simd_float4x4 m = {};
    m.columns[0].x = f / aspect;
    m.columns[1].y = f;
    m.columns[2].z = far / (near - far);
    m.columns[2].w = -1.0f;
    m.columns[3].z = far * near / (near - far);
    return m;
}

// Uniform buffers matching shader structs.
// Use float4 columns for normal matrix to ensure 16-byte alignment.
struct Uniforms {
    simd_float4x4 mvp;
    simd_float4x4 model_mat;
    simd_float4   normal_col0;  // normal matrix column 0 (w unused)
    simd_float4   normal_col1;  // normal matrix column 1 (w unused)
    simd_float4   normal_col2;  // normal matrix column 2 (w unused)
    simd_float4   color;
};

struct SceneUniforms {
    simd_float4x4 view_proj;
    simd_float3   light_dir;
    float          _pad0;
    simd_float3   light_dir2;
    float          _pad1;
    simd_float3   eye_pos;
    float          _pad2;
};

struct FloorUniforms {
    simd_float4x4 mvp;
    float          half_size;
    int            grid_divs;
    float          _pad[2];
};

void mujoco_renderer_render(MujocoRenderer *r, MujocoContext *mj,
                            mjvCamera *cam,
                            bool show_skin, bool show_bodies,
                            bool show_sites, bool show_arena,
                            const ViewOverride *view_override,
                            bool show_arena_corners,
                            void *bg_texture,
                            float scene_opacity,
                            float bg_zoom,
                            const float *bg_pan,
                            float arena_width,
                            float arena_depth,
                            const float *arena_offset) {
    if (!r || !mj || !mj->loaded || !cam) return;

    @autoreleasepool {
        // Update abstract scene using the MuJoCo camera
        mjv_updateScene(mj->model, mj->data, &mj->opt, nullptr, cam, mjCAT_ALL, &mj->scene);

        // Build view/projection matrices
        simd_float4x4 vp;
        simd_float3 eye;

        if (view_override && view_override->active) {
            // Direct view/projection from calibration camera extrinsics
            simd_float4x4 view, proj;
            memcpy(&view, view_override->view, sizeof(view));
            memcpy(&proj, view_override->proj, sizeof(proj));
            vp = simd_mul(proj, view);
            eye = {view_override->eye[0], view_override->eye[1], view_override->eye[2]};
        } else {
            // Derive from mjvCamera spherical coordinates
            // Matches MuJoCo's mjv_cameraFrame (engine_vis_visualize.c):
            //   forward = { ce*ca, ce*sa, se }
            //   up      = { -se*ca, -se*sa, ce }
            //   eye     = lookat - distance * forward
            float az = (float)cam->azimuth * M_PI / 180.0f;
            float el = (float)cam->elevation * M_PI / 180.0f;
            float ca_ = cosf(az), sa_ = sinf(az);
            float ce_ = cosf(el), se_ = sinf(el);
            float dist = (float)cam->distance;
            simd_float3 center = {(float)cam->lookat[0], (float)cam->lookat[1], (float)cam->lookat[2]};
            eye = {
                center.x - dist * ce_ * ca_,
                center.y - dist * ce_ * sa_,
                center.z - dist * se_
            };
            simd_float3 up = {-se_ * ca_, -se_ * sa_, ce_};

            float aspect = (float)r->width / (float)r->height;
            simd_float4x4 view = look_at(eye, center, up);
            simd_float4x4 proj = perspective(45.0f * M_PI / 180.0f, aspect, 0.001f, 100.0f);
            vp = simd_mul(proj, view);
        }

        SceneUniforms su;
        su.view_proj = vp;
        su.light_dir  = simd_normalize((simd_float3){0.2f, 0.1f, 1.0f});  // key: above
        su.light_dir2 = simd_normalize((simd_float3){0.5f, 0.8f, 0.3f});  // fill: side
        su.eye_pos = eye;

        // Begin render pass
        id<MTLCommandBuffer> cmd = [r->queue commandBuffer];
        MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = r->color_tex;
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].clearColor = bg_texture
            ? MTLClearColorMake(0, 0, 0, 0)  // transparent: let video show through
            : MTLClearColorMake(0.15, 0.15, 0.18, 1.0);
        rpd.depthAttachment.texture = r->depth_tex;
        rpd.depthAttachment.loadAction = MTLLoadActionClear;
        rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
        rpd.depthAttachment.clearDepth = 1.0;

        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];

        // Draw video background if provided (before 3D scene, no depth write)
        if (bg_texture && r->blit_pipeline) {
            id<MTLTexture> bg_tex = (__bridge id<MTLTexture>)(bg_texture);
            // Pack zoom/pan for the blit shader
            struct { simd_float2 uv_scale; simd_float2 uv_offset; } blit_params;
            blit_params.uv_scale = {bg_zoom, bg_zoom};
            blit_params.uv_offset = bg_pan ? (simd_float2){bg_pan[0], bg_pan[1]}
                                           : (simd_float2){0, 0};
            [enc setRenderPipelineState:r->blit_pipeline];
            [enc setVertexBytes:&blit_params length:sizeof(blit_params) atIndex:0];
            [enc setFragmentTexture:bg_tex atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }

        [enc setRenderPipelineState:r->pipeline];
        [enc setDepthStencilState:r->depth_state];
        [enc setCullMode:MTLCullModeBack];
        [enc setFrontFacingWinding:MTLWindingCounterClockwise];

        // Scene-level uniforms
        [enc setVertexBytes:&su length:sizeof(su) atIndex:2];
        [enc setFragmentBytes:&su length:sizeof(su) atIndex:2];

        // Draw each geom
        for (int i = 0; i < mj->scene.ngeom; i++) {
            const mjvGeom &g = mj->scene.geoms[i];
            if (g.rgba[3] < 0.01f) continue; // invisible
            if (!show_sites && g.objtype == mjOBJ_SITE) continue;
            if (!show_bodies && g.objtype != mjOBJ_SITE) continue;

            id<MTLBuffer> vb = nil, ib = nil;
            int idx_count = 0;
            simd_float4x4 model_mat;

            // Helper: emit one draw call
            auto draw_geom = [&](id<MTLBuffer> vb, id<MTLBuffer> ib,
                                 int idx_count, simd_float4x4 model_mat) {
                Uniforms u;
                u.model_mat = model_mat;
                u.mvp = simd_mul(vp, model_mat);
                extract_normal_columns(model_mat, u.normal_col0, u.normal_col1, u.normal_col2);
                // Force full opacity for body geoms (mjvScene may set alpha < 1
                // for certain geom groups, causing see-through artifacts)
                float alpha = (g.objtype == mjOBJ_SITE) ? g.rgba[3] : 1.0f;
                u.color = {g.rgba[0], g.rgba[1], g.rgba[2], alpha * scene_opacity};
                [enc setVertexBuffer:vb offset:0 atIndex:0];
                [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:idx_count indexType:MTLIndexTypeUInt32
                               indexBuffer:ib indexBufferOffset:0];
            };

            switch (g.type) {
                case mjGEOM_SPHERE:
                case mjGEOM_ELLIPSOID: {
                    float sx = g.size[0], sy = g.size[1], sz = g.size[2];
                    if (g.type == mjGEOM_SPHERE) sy = sz = sx;
                    draw_geom(r->sphere_vb, r->sphere_ib, r->sphere_idx_count,
                              geom_model_matrix(g, sx, sy, sz));
                    break;
                }
                case mjGEOM_CAPSULE: {
                    float radius = g.size[0];
                    float half_len = g.size[2];
                    draw_geom(r->capsule_vb, r->capsule_ib, r->capsule_idx_count,
                              geom_model_matrix(g, radius, radius, half_len));
                    break;
                }
                case mjGEOM_CYLINDER: {
                    float radius = g.size[0];
                    float half_len = g.size[2];
                    draw_geom(r->cylinder_vb, r->cylinder_ib, r->cylinder_idx_count,
                              geom_model_matrix(g, radius, radius, half_len));
                    break;
                }
                case mjGEOM_BOX: {
                    draw_geom(r->box_vb, r->box_ib, r->box_idx_count,
                              geom_model_matrix(g, g.size[0], g.size[1], g.size[2]));
                    break;
                }
                case mjGEOM_MESH: {
                    // Mesh geom: dataid encodes 2*meshid or 2*meshid+1 (hull)
                    if (g.dataid < 0) continue;
                    int meshid = g.dataid / 2;
                    if (meshid >= mj->model->nmesh) continue;

                    int nface = mj->model->mesh_facenum[meshid];
                    if (nface == 0) continue;

                    int vertadr = mj->model->mesh_vertadr[meshid];
                    int normaladr = mj->model->mesh_normaladr[meshid];
                    int fadr = mj->model->mesh_faceadr[meshid];
                    int *fv = mj->model->mesh_face + fadr * 3;
                    int *fn = mj->model->mesh_facenormal + fadr * 3;

                    // Build per-face-vertex buffer (non-indexed), matching MuJoCo's
                    // native GL upload: vertices and normals use SEPARATE index
                    // arrays (mesh_face vs mesh_facenormal), so we can't use indexed
                    // drawing. Each triangle gets 3 unique vertices with correct normals.
                    int ntri = nface;
                    size_t vb_size = ntri * 3 * sizeof(Vertex);
                    id<MTLBuffer> mesh_vb = [r->device newBufferWithLength:vb_size
                                             options:MTLResourceStorageModeShared];
                    Vertex *verts = (Vertex *)[mesh_vb contents];
                    for (int f = 0; f < ntri; f++) {
                        for (int c = 0; c < 3; c++) {
                            int vi = fv[3*f + c] + vertadr;
                            int ni = fn[3*f + c] + normaladr;
                            verts[3*f + c].position = {
                                mj->model->mesh_vert[3*vi],
                                mj->model->mesh_vert[3*vi+1],
                                mj->model->mesh_vert[3*vi+2]};
                            verts[3*f + c].normal = {
                                mj->model->mesh_normal[3*ni],
                                mj->model->mesh_normal[3*ni+1],
                                mj->model->mesh_normal[3*ni+2]};
                        }
                    }

                    // Draw as non-indexed triangles with backface culling ON
                    // (same as MuJoCo GL — resolves z-fighting between shells)
                    Uniforms u;
                    simd_float4x4 model_mat = geom_model_matrix(g, 1.0f, 1.0f, 1.0f);
                    u.model_mat = model_mat;
                    u.mvp = simd_mul(vp, model_mat);
                    extract_normal_columns(model_mat, u.normal_col0, u.normal_col1, u.normal_col2);
                    float a = (g.objtype == mjOBJ_SITE) ? g.rgba[3] : 1.0f;
                    u.color = {g.rgba[0], g.rgba[1], g.rgba[2], a * scene_opacity};
                    [enc setVertexBuffer:mesh_vb offset:0 atIndex:0];
                    [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
                    [enc drawPrimitives:MTLPrimitiveTypeTriangle
                            vertexStart:0 vertexCount:ntri * 3];
                    break;
                }
                default:
                    continue;
            }
        }

        // --- Render skins (smooth mesh) ---
        for (int si = 0; show_skin && si < mj->scene.nskin; si++) {
            int nvert = mj->scene.skinvertnum[si];
            int nface = mj->scene.skinfacenum[si];
            if (nvert == 0 || nface == 0) continue;

            // Build vertex buffer from scene skin data (updated each frame by mjv_updateScene)
            int vadr = mj->scene.skinvertadr[si];
            float *vpos = mj->scene.skinvert + vadr * 3;
            float *vnorm = mj->scene.skinnormal + vadr * 3;

            // Interleave position + normal into Vertex structs
            size_t vb_size = nvert * sizeof(Vertex);
            if (!r->skin_vb || [r->skin_vb length] < vb_size) {
                r->skin_vb = [r->device newBufferWithLength:vb_size options:MTLResourceStorageModeShared];
            }
            Vertex *verts = (Vertex *)[r->skin_vb contents];
            for (int v = 0; v < nvert; v++) {
                verts[v].position = {vpos[3*v], vpos[3*v+1], vpos[3*v+2]};
                verts[v].normal = {vnorm[3*v], vnorm[3*v+1], vnorm[3*v+2]};
            }

            // Build index buffer from model face data (static, only once)
            if (!r->skin_ib_built || !r->skin_ib) {
                // Face indices come from mjModel, not mjvScene
                int *faces = mj->model->skin_face + mj->model->skin_faceadr[si] * 3;
                size_t ib_size = nface * 3 * sizeof(uint32_t);
                std::vector<uint32_t> idx(nface * 3);
                for (int f = 0; f < nface * 3; f++) idx[f] = (uint32_t)faces[f];
                r->skin_ib = make_buffer(r->device, idx.data(), ib_size);
                r->skin_face_count = nface;
                r->skin_ib_built = true;
            }

            // Draw skin as identity transform (verts are already in world space)
            Uniforms u;
            u.model_mat = matrix_identity_float4x4;
            u.mvp = vp; // just view-proj, no model transform
            u.normal_col0 = {1, 0, 0, 0};
            u.normal_col1 = {0, 1, 0, 0};
            u.normal_col2 = {0, 0, 1, 0};
            u.color = {0.45f, 0.75f, 0.85f, 0.85f * scene_opacity};

            [enc setVertexBuffer:r->skin_vb offset:0 atIndex:0];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:r->skin_face_count * 3
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:r->skin_ib
                     indexBufferOffset:0];
        }

        // --- Render arena (checkerboard floor) ---
        if (show_arena) {
            float hw = arena_width * 0.5f;
            float hd = arena_depth * 0.5f;
            float ox = arena_offset ? arena_offset[0] : 0.0f;
            float oy = arena_offset ? arena_offset[1] : 0.0f;
            float oz = arena_offset ? arena_offset[2] : 0.0f;

            // Build floor quad centered at arena_offset
            Vertex floor_verts[4] = {
                {{ox-hw, oy-hd, oz}, {0, 0, 1}},
                {{ox+hw, oy-hd, oz}, {0, 0, 1}},
                {{ox+hw, oy+hd, oz}, {0, 0, 1}},
                {{ox-hw, oy+hd, oz}, {0, 0, 1}},
            };
            uint32_t floor_idx[6] = {0, 1, 2, 0, 2, 3};
            id<MTLBuffer> fvb = [r->device newBufferWithBytes:floor_verts
                                 length:sizeof(floor_verts) options:MTLResourceStorageModeShared];
            id<MTLBuffer> fib = [r->device newBufferWithBytes:floor_idx
                                 length:sizeof(floor_idx) options:MTLResourceStorageModeShared];

            [enc setRenderPipelineState:r->floor_pipeline];

            Uniforms fu;
            fu.model_mat = matrix_identity_float4x4;
            fu.mvp = vp;
            fu.normal_col0 = {1,0,0,0}; fu.normal_col1 = {0,1,0,0}; fu.normal_col2 = {0,0,1,0};
            fu.color = {0.25f, 0.32f, 0.42f, scene_opacity};

            FloorUniforms ffu;
            ffu.mvp = vp;
            ffu.half_size = std::max(hw, hd); // used for UV scaling in shader
            ffu.grid_divs = 4;

            [enc setVertexBuffer:fvb offset:0 atIndex:0];
            [enc setVertexBytes:&fu length:sizeof(fu) atIndex:1];
            [enc setFragmentBytes:&ffu length:sizeof(ffu) atIndex:3];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:6
                             indexType:MTLIndexTypeUInt32 indexBuffer:fib indexBufferOffset:0];

            [enc setRenderPipelineState:r->pipeline];
        }

        // --- Render arena corner markers during alignment mode ---
        if (show_arena_corners) {
            float hw = arena_width * 0.5f;
            float hd = arena_depth * 0.5f;
            float ox = arena_offset ? arena_offset[0] : 0.0f;
            float oy = arena_offset ? arena_offset[1] : 0.0f;
            float oz = arena_offset ? arena_offset[2] : 0.0f;
            float corner_pos[4][3] = {
                {ox+hw, oy+hd, oz+0.005f},
                {ox-hw, oy+hd, oz+0.005f},
                {ox-hw, oy-hd, oz+0.005f},
                {ox+hw, oy-hd, oz+0.005f},
            };
            // Colors matching ArenaCorners4 skeleton: red, green, blue, yellow
            float corner_colors[4][4] = {
                {1.0f, 0.2f, 0.2f, 1.0f},
                {0.2f, 0.9f, 0.2f, 1.0f},
                {0.3f, 0.5f, 1.0f, 1.0f},
                {1.0f, 0.9f, 0.1f, 1.0f},
            };
            float corner_r = 0.025f; // 25mm radius spheres

            for (int ci = 0; ci < 4; ci++) {
                simd_float4x4 model_mat;
                model_mat.columns[0] = {corner_r, 0, 0, 0};
                model_mat.columns[1] = {0, corner_r, 0, 0};
                model_mat.columns[2] = {0, 0, corner_r, 0};
                model_mat.columns[3] = {corner_pos[ci][0], corner_pos[ci][1], corner_pos[ci][2], 1};

                Uniforms u;
                u.model_mat = model_mat;
                u.mvp = simd_mul(vp, model_mat);
                extract_normal_columns(model_mat, u.normal_col0, u.normal_col1, u.normal_col2);
                u.color = {corner_colors[ci][0], corner_colors[ci][1],
                           corner_colors[ci][2], corner_colors[ci][3] * scene_opacity};

                [enc setVertexBuffer:r->sphere_vb offset:0 atIndex:0];
                [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
                [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                indexCount:r->sphere_idx_count indexType:MTLIndexTypeUInt32
                               indexBuffer:r->sphere_ib indexBufferOffset:0];
            }
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

ImTextureID mujoco_renderer_get_texture(MujocoRenderer *r) {
    if (!r || !r->color_tex) return (ImTextureID)0;
    return (ImTextureID)(intptr_t)(__bridge void *)(r->color_tex);
}

void mujoco_renderer_get_size(MujocoRenderer *r, uint32_t *w, uint32_t *h) {
    if (r) { *w = r->width; *h = r->height; }
    else   { *w = 0; *h = 0; }
}

#endif // RED_HAS_MUJOCO
#endif // __APPLE__
