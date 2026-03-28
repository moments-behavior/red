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
    float3   eye_pos;
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
                              constant SceneUniforms &su [[buffer(2)]]) {
    float3 N = normalize(in.normal);
    float3 L = normalize(su.light_dir);
    float3 V = normalize(su.eye_pos - in.world_pos);
    float3 H = normalize(L + V);

    float ambient = 0.25;
    float diffuse = max(dot(N, L), 0.0) * 0.55;
    float spec = pow(max(dot(N, H), 0.0), 32.0) * 0.2;

    float3 color = in.color.rgb * (ambient + diffuse) + float3(spec);
    return float4(color, in.color.a);
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
            simd_float3 n = {sp * ct, cp, sp * st};
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
    int total_rings = rings + 2; // hemisphere + 2 cylinder + hemisphere
    int stride = sectors + 1;
    for (int r = 0; r < total_rings; r++) {
        for (int s = 0; s < sectors; s++) {
            uint32_t a = r * stride + s;
            uint32_t b = a + stride;
            idx.insert(idx.end(), {a, b, a + 1, a + 1, b, b + 1});
        }
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
        r->skin_vb = nil;    r->skin_ib = nil;
        r->pipeline = nil;
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

static simd_float4x4 perspective(float fov_y, float aspect, float near, float far) {
    float f = 1.0f / tanf(fov_y * 0.5f);
    simd_float4x4 m = {};
    m.columns[0].x = f / aspect;
    m.columns[1].y = f;
    m.columns[2].z = (far + near) / (near - far);
    m.columns[2].w = -1.0f;
    m.columns[3].z = 2.0f * far * near / (near - far);
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
    simd_float3   eye_pos;
    float          _pad1;
};

void mujoco_renderer_render(MujocoRenderer *r, MujocoContext *mj,
                            const float lookat[3], float distance,
                            float azimuth, float elevation,
                            bool show_skin, bool show_sites) {
    if (!r || !mj || !mj->loaded) return;

    @autoreleasepool {
        // Update abstract scene
        mjvCamera cam;
        mjv_defaultCamera(&cam);
        cam.lookat[0] = lookat[0];
        cam.lookat[1] = lookat[1];
        cam.lookat[2] = lookat[2];
        cam.distance = distance;
        cam.azimuth = azimuth;
        cam.elevation = elevation;
        mjv_updateScene(mj->model, mj->data, &mj->opt, nullptr, &cam, mjCAT_ALL, &mj->scene);

        // Build view/projection
        float az = azimuth * M_PI / 180.0f;
        float el = elevation * M_PI / 180.0f;
        simd_float3 eye = {
            lookat[0] + distance * cosf(el) * sinf(az),
            lookat[1] + distance * sinf(el),
            lookat[2] + distance * cosf(el) * cosf(az)
        };
        simd_float3 center = {lookat[0], lookat[1], lookat[2]};
        simd_float3 up = {0, 1, 0};

        float aspect = (float)r->width / (float)r->height;
        simd_float4x4 view = look_at(eye, center, up);
        simd_float4x4 proj = perspective(45.0f * M_PI / 180.0f, aspect, 0.001f, 100.0f);
        simd_float4x4 vp = simd_mul(proj, view);

        SceneUniforms su;
        su.view_proj = vp;
        su.light_dir = simd_normalize((simd_float3){0.5f, 1.0f, 0.3f});
        su.eye_pos = eye;

        // Begin render pass
        id<MTLCommandBuffer> cmd = [r->queue commandBuffer];
        MTLRenderPassDescriptor *rpd = [MTLRenderPassDescriptor renderPassDescriptor];
        rpd.colorAttachments[0].texture = r->color_tex;
        rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
        rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
        rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.15, 0.15, 0.18, 1.0);
        rpd.depthAttachment.texture = r->depth_tex;
        rpd.depthAttachment.loadAction = MTLLoadActionClear;
        rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
        rpd.depthAttachment.clearDepth = 1.0;

        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rpd];
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
            if (!show_sites && g.objtype == mjOBJ_SITE) continue; // hide sites

            id<MTLBuffer> vb = nil, ib = nil;
            int idx_count = 0;
            simd_float4x4 model_mat;

            switch (g.type) {
                case mjGEOM_SPHERE:
                case mjGEOM_ELLIPSOID: {
                    vb = r->sphere_vb; ib = r->sphere_ib;
                    idx_count = r->sphere_idx_count;
                    float sx = g.size[0], sy = g.size[1], sz = g.size[2];
                    if (g.type == mjGEOM_SPHERE) sy = sz = sx;
                    model_mat = geom_model_matrix(g, sx, sy, sz);
                    break;
                }
                case mjGEOM_CAPSULE: {
                    vb = r->capsule_vb; ib = r->capsule_ib;
                    idx_count = r->capsule_idx_count;
                    float radius = g.size[0];
                    float half_len = g.size[2]; // mjvGeom: size=(radius, radius, half_cyl_len)
                    // MuJoCo capsule axis = Z of geom frame.
                    // Unit capsule extends along Z: X=radius, Y=radius, Z=half_len.
                    model_mat = geom_model_matrix(g, radius, radius, half_len);
                    break;
                }
                case mjGEOM_CYLINDER: {
                    vb = r->capsule_vb; ib = r->capsule_ib;
                    idx_count = r->capsule_idx_count;
                    float radius = g.size[0];
                    float half_len = g.size[2]; // mjvGeom: size=(radius, radius, half_cyl_len)
                    model_mat = geom_model_matrix(g, radius, radius, half_len);
                    break;
                }
                case mjGEOM_BOX: {
                    vb = r->box_vb; ib = r->box_ib;
                    idx_count = r->box_idx_count;
                    model_mat = geom_model_matrix(g, g.size[0], g.size[1], g.size[2]);
                    break;
                }
                default:
                    // Skip mesh, plane, etc. for now
                    continue;
            }

            if (!vb || !ib || idx_count == 0) continue;

            Uniforms u;
            u.model_mat = model_mat;
            u.mvp = simd_mul(vp, model_mat);
            extract_normal_columns(model_mat, u.normal_col0, u.normal_col1, u.normal_col2);
            u.color = {g.rgba[0], g.rgba[1], g.rgba[2], g.rgba[3]};

            [enc setVertexBuffer:vb offset:0 atIndex:0];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:idx_count
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:ib
                     indexBufferOffset:0];
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
            u.color = {0.45f, 0.75f, 0.85f, 0.85f}; // light blue, slightly transparent

            [enc setVertexBuffer:r->skin_vb offset:0 atIndex:0];
            [enc setVertexBytes:&u length:sizeof(u) atIndex:1];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                            indexCount:r->skin_face_count * 3
                             indexType:MTLIndexTypeUInt32
                           indexBuffer:r->skin_ib
                     indexBufferOffset:0];
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
