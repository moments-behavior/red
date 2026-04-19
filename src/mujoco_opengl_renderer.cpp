#ifndef __APPLE__
#ifdef RED_HAS_MUJOCO

#include "mujoco_opengl_renderer.h"
#include "mujoco_context.h"
#include <mujoco.h>
#include <GL/glew.h>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// GLSL Shader sources
// ---------------------------------------------------------------------------

static const char *kVertexShaderSrc = R"(
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;

uniform mat4 uMVP;
uniform mat4 uModelMat;
uniform mat3 uNormalMat;
uniform vec4 uColor;

out vec3 vNormal;
out vec3 vWorldPos;
out vec4 vColor;

void main() {
    vec4 wp = uModelMat * vec4(aPosition, 1.0);
    vWorldPos = wp.xyz;
    gl_Position = uMVP * vec4(aPosition, 1.0);
    vNormal = normalize(uNormalMat * aNormal);
    vColor = uColor;
}
)";

static const char *kFragmentShaderSrc = R"(
#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;
in vec4 vColor;

uniform vec3 uLightDir;
uniform vec3 uLightDir2;
uniform vec3 uEyePos;
uniform float uBrightness;
uniform float uContrast;

out vec4 FragColor;

void main() {
    vec3 N = normalize(vNormal);
    if (!gl_FrontFacing) N = -N;
    vec3 V = normalize(uEyePos - vWorldPos);

    vec3 L1 = normalize(uLightDir);
    vec3 L2 = normalize(uLightDir2);
    vec3 H1 = normalize(L1 + V);
    vec3 H2 = normalize(L2 + V);

    float ambient = 0.35;
    float diffuse = max(dot(N, L1), 0.0) * 0.50
                  + max(dot(N, L2), 0.0) * 0.30;
    float spec = pow(max(dot(N, H1), 0.0), 32.0) * 0.20
               + pow(max(dot(N, H2), 0.0), 32.0) * 0.10;

    vec3 color = vColor.rgb * (ambient + diffuse) + vec3(spec);
    color = (color - 0.5) * uContrast + 0.5 + uBrightness;
    color = clamp(color, 0.0, 1.0);
    FragColor = vec4(color, vColor.a);
}
)";

static const char *kFloorFragmentShaderSrc = R"(
#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;
in vec4 vColor;

uniform vec3 uLightDir;
uniform vec3 uLightDir2;
uniform vec3 uEyePos;
uniform float uBrightness;
uniform float uContrast;
uniform float uFloorHalfSize;
uniform int uFloorGridDivs;

out vec4 FragColor;

void main() {
    vec3 N = vec3(0.0, 0.0, 1.0);
    vec3 V = normalize(uEyePos - vWorldPos);

    float tile_size = (2.0 * uFloorHalfSize) / float(uFloorGridDivs);
    vec2 uv = (vWorldPos.xy + vec2(uFloorHalfSize)) / tile_size;
    ivec2 tile = ivec2(floor(uv));
    vec2 frac_uv = fract(uv);

    bool checker = ((tile.x + tile.y) & 1) == 0;
    vec3 tile_dark  = vec3(0.22, 0.28, 0.36);
    vec3 tile_light = vec3(0.28, 0.35, 0.44);
    vec3 base_color = checker ? tile_dark : tile_light;

    float line_w = 0.015;
    float lx = min(frac_uv.x, 1.0 - frac_uv.x);
    float ly = min(frac_uv.y, 1.0 - frac_uv.y);
    float line_dist = min(lx, ly);
    float line = 1.0 - smoothstep(0.0, line_w, line_dist);
    vec3 line_color = vec3(0.75, 0.80, 0.85);
    base_color = mix(base_color, line_color, line * 0.8);

    vec3 L1 = normalize(uLightDir);
    vec3 L2 = normalize(uLightDir2);
    vec3 H1 = normalize(L1 + V);
    vec3 H2 = normalize(L2 + V);

    float ambient = 0.25;
    float diffuse = max(dot(N, L1), 0.0) * 0.45
                  + max(dot(N, L2), 0.0) * 0.20;
    float spec = pow(max(dot(N, H1), 0.0), 64.0) * 0.45
               + pow(max(dot(N, H2), 0.0), 64.0) * 0.15;

    vec3 color = base_color * (ambient + diffuse) + vec3(spec);
    color = (color - 0.5) * uContrast + 0.5 + uBrightness;
    color = clamp(color, 0.0, 1.0);
    FragColor = vec4(color, vColor.a);
}
)";

static const char *kBlitVertexShaderSrc = R"(
#version 330 core
uniform vec2 uUVScale;
uniform vec2 uUVOffset;

out vec2 vUV;

void main() {
    // Fullscreen triangle: 3 vertices covering the screen
    vec2 base_uv = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(base_uv * 2.0 - 1.0, 0.0, 1.0);
    vec2 uv = base_uv;
    uv.y = 1.0 - uv.y;
    uv = (uv - 0.5) * uUVScale + 0.5 + uUVOffset;
    vUV = uv;
}
)";

static const char *kBlitFragmentShaderSrc = R"(
#version 330 core
in vec2 vUV;
uniform sampler2D uTexture;
out vec4 FragColor;

void main() {
    if (vUV.x < 0.0 || vUV.x > 1.0 || vUV.y < 0.0 || vUV.y > 1.0) {
        FragColor = vec4(0, 0, 0, 1);
        return;
    }
    FragColor = texture(uTexture, vUV);
}
)";

// ---------------------------------------------------------------------------
// Shader compilation helpers
// ---------------------------------------------------------------------------

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "[MuJoCo GL] Shader compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs) {
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof(log), nullptr, log);
        fprintf(stderr, "[MuJoCo GL] Program link error: %s\n", log);
        glDeleteProgram(p);
        return 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return p;
}

// ---------------------------------------------------------------------------
// Unit geometry generators
// ---------------------------------------------------------------------------

struct Vertex {
    float position[3];
    float normal[3];
};

static std::vector<Vertex> make_unit_sphere(int rings, int sectors) {
    std::vector<Vertex> verts;
    for (int r = 0; r <= rings; r++) {
        float phi = (float)M_PI * r / rings;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * (float)M_PI * s / sectors;
            float sp = sinf(phi), cp = cosf(phi);
            float st = sinf(theta), ct = cosf(theta);
            float n[3] = {sp * ct, sp * st, cp};
            verts.push_back({n[0], n[1], n[2], n[0], n[1], n[2]});
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

static std::vector<Vertex> make_unit_capsule(int rings, int sectors) {
    std::vector<Vertex> verts;
    // Top hemisphere
    for (int r = 0; r <= rings / 2; r++) {
        float phi = (float)M_PI * r / rings;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * (float)M_PI * s / sectors;
            float sp = sinf(phi), cp = cosf(phi);
            float st = sinf(theta), ct = cosf(theta);
            float n[3] = {sp * ct, sp * st, cp};
            float p[3] = {n[0], n[1], n[2] + 0.5f};
            verts.push_back({p[0], p[1], p[2], n[0], n[1], n[2]});
        }
    }
    // Cylinder body
    for (int cap = 0; cap < 2; cap++) {
        float z = cap ? -0.5f : 0.5f;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * (float)M_PI * s / sectors;
            float n[3] = {cosf(theta), sinf(theta), 0.0f};
            verts.push_back({n[0], n[1], z, n[0], n[1], 0.0f});
        }
    }
    // Bottom hemisphere
    for (int r = rings / 2; r <= rings; r++) {
        float phi = (float)M_PI * r / rings;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * (float)M_PI * s / sectors;
            float sp = sinf(phi), cp = cosf(phi);
            float st = sinf(theta), ct = cosf(theta);
            float n[3] = {sp * ct, sp * st, cp};
            float p[3] = {n[0], n[1], n[2] - 0.5f};
            verts.push_back({p[0], p[1], p[2], n[0], n[1], n[2]});
        }
    }
    return verts;
}

static std::vector<uint32_t> make_capsule_indices(int rings, int sectors) {
    std::vector<uint32_t> idx;
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

static std::vector<Vertex> make_unit_cylinder(int sectors) {
    std::vector<Vertex> verts;
    for (int cap = 0; cap < 2; cap++) {
        float z = cap ? -1.0f : 1.0f;
        for (int s = 0; s <= sectors; s++) {
            float theta = 2.0f * (float)M_PI * s / sectors;
            float n[3] = {cosf(theta), sinf(theta), 0.0f};
            verts.push_back({n[0], n[1], z, n[0], n[1], 0.0f});
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

static std::vector<Vertex> make_unit_box() {
    std::vector<Vertex> v;
    auto face = [&](float nx, float ny, float nz,
                     float ux, float uy, float uz,
                     float rx, float ry, float rz) {
        float p0[3] = {nx+ux+rx, ny+uy+ry, nz+uz+rz};
        float p1[3] = {nx-ux+rx, ny-uy+ry, nz-uz+rz};
        float p2[3] = {nx-ux-rx, ny-uy-ry, nz-uz-rz};
        float p3[3] = {nx+ux-rx, ny+uy-ry, nz+uz-rz};
        v.push_back({p0[0],p0[1],p0[2], nx,ny,nz});
        v.push_back({p1[0],p1[1],p1[2], nx,ny,nz});
        v.push_back({p2[0],p2[1],p2[2], nx,ny,nz});
        v.push_back({p3[0],p3[1],p3[2], nx,ny,nz});
    };
    face( 0, 0, 1,  1, 0, 0,  0, 1, 0);  // +Z
    face( 0, 0,-1, -1, 0, 0,  0, 1, 0);  // -Z
    face( 1, 0, 0,  0, 0, 1,  0, 1, 0);  // +X
    face(-1, 0, 0,  0, 0,-1,  0, 1, 0);  // -X
    face( 0, 1, 0,  1, 0, 0,  0, 0, 1);  // +Y
    face( 0,-1, 0,  1, 0, 0,  0, 0,-1);  // -Y
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
// Geometry buffer (VAO + VBO + EBO)
// ---------------------------------------------------------------------------

struct GeomBuffer {
    GLuint vao = 0, vbo = 0, ebo = 0;
    int index_count = 0;
};

static GeomBuffer create_geom_buffer(const std::vector<Vertex> &verts,
                                      const std::vector<uint32_t> &indices) {
    GeomBuffer gb;
    gb.index_count = (int)indices.size();

    glGenVertexArrays(1, &gb.vao);
    glGenBuffers(1, &gb.vbo);
    glGenBuffers(1, &gb.ebo);

    glBindVertexArray(gb.vao);

    glBindBuffer(GL_ARRAY_BUFFER, gb.vbo);
    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gb.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(uint32_t), indices.data(), GL_STATIC_DRAW);

    // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)0);
    // Normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void *)(3 * sizeof(float)));

    glBindVertexArray(0);
    return gb;
}

static void destroy_geom_buffer(GeomBuffer &gb) {
    if (gb.vao) glDeleteVertexArrays(1, &gb.vao);
    if (gb.vbo) glDeleteBuffers(1, &gb.vbo);
    if (gb.ebo) glDeleteBuffers(1, &gb.ebo);
    gb = {};
}

// ---------------------------------------------------------------------------
// Mesh cache entry (non-indexed triangles — matches Metal renderer)
// ---------------------------------------------------------------------------

struct MeshCacheEntry {
    GLuint vao = 0, vbo = 0;
    int tri_count = 0;
};

// ---------------------------------------------------------------------------
// Renderer state
// ---------------------------------------------------------------------------

struct MujocoRenderer {
    uint32_t width, height;

    // FBO
    GLuint fbo = 0;
    GLuint color_tex = 0;
    GLuint depth_rb = 0;

    // Shader programs
    GLuint prog_main = 0;
    GLuint prog_floor = 0;
    GLuint prog_blit = 0;

    // Uniform locations — main program
    GLint loc_mvp, loc_model_mat, loc_normal_mat, loc_color;
    GLint loc_light_dir, loc_light_dir2, loc_eye_pos;
    GLint loc_brightness, loc_contrast;

    // Uniform locations — floor program
    GLint floc_mvp, floc_model_mat, floc_normal_mat, floc_color;
    GLint floc_light_dir, floc_light_dir2, floc_eye_pos;
    GLint floc_brightness, floc_contrast;
    GLint floc_half_size, floc_grid_divs;

    // Uniform locations — blit program
    GLint bloc_uv_scale, bloc_uv_offset, bloc_texture;

    // Blit VAO (empty, uses gl_VertexID)
    GLuint blit_vao = 0;

    // Unit geometry
    GeomBuffer sphere, capsule, box, cylinder;

    // Mesh cache
    std::unordered_map<int, MeshCacheEntry> mesh_cache;

    // Skin buffers
    GLuint skin_vao = 0, skin_vbo = 0, skin_ebo = 0;
    int skin_vert_count = 0;
    int skin_face_count = 0;
    bool skin_ebo_built = false;
};

// ---------------------------------------------------------------------------
// FBO creation
// ---------------------------------------------------------------------------

static void create_fbo(MujocoRenderer *r) {
    // Color texture
    glGenTextures(1, &r->color_tex);
    glBindTexture(GL_TEXTURE_2D, r->color_tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, r->width, r->height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Depth renderbuffer
    glGenRenderbuffers(1, &r->depth_rb);
    glBindRenderbuffer(GL_RENDERBUFFER, r->depth_rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, r->width, r->height);

    // FBO
    glGenFramebuffers(1, &r->fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, r->fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, r->color_tex, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, r->depth_rb);

    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[MuJoCo GL] FBO incomplete: 0x%x\n", status);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

static void destroy_fbo(MujocoRenderer *r) {
    if (r->color_tex) glDeleteTextures(1, &r->color_tex);
    if (r->depth_rb) glDeleteRenderbuffers(1, &r->depth_rb);
    if (r->fbo) glDeleteFramebuffers(1, &r->fbo);
    r->color_tex = 0;
    r->depth_rb = 0;
    r->fbo = 0;
}

// ---------------------------------------------------------------------------
// Matrix math (column-major, matching OpenGL conventions)
// ---------------------------------------------------------------------------

struct Mat4 {
    float m[16]; // column-major
    float &operator()(int row, int col) { return m[col * 4 + row]; }
    float operator()(int row, int col) const { return m[col * 4 + row]; }
};

static Mat4 mat4_identity() {
    Mat4 r = {};
    r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0f;
    return r;
}

static Mat4 mat4_mul(const Mat4 &a, const Mat4 &b) {
    Mat4 r = {};
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++)
            for (int k = 0; k < 4; k++)
                r(row, c) += a(row, k) * b(k, c);
    return r;
}

static Mat4 look_at(const float *eye, const float *center, const float *up) {
    float fx = center[0] - eye[0], fy = center[1] - eye[1], fz = center[2] - eye[2];
    float len = sqrtf(fx*fx + fy*fy + fz*fz);
    fx /= len; fy /= len; fz /= len;
    // s = f x up
    float sx = fy*up[2] - fz*up[1], sy = fz*up[0] - fx*up[2], sz = fx*up[1] - fy*up[0];
    len = sqrtf(sx*sx + sy*sy + sz*sz);
    sx /= len; sy /= len; sz /= len;
    // u = s x f
    float ux = sy*fz - sz*fy, uy = sz*fx - sx*fz, uz = sx*fy - sy*fx;

    Mat4 m = {};
    m(0,0) = sx;  m(0,1) = sy;  m(0,2) = sz;  m(0,3) = -(sx*eye[0] + sy*eye[1] + sz*eye[2]);
    m(1,0) = ux;  m(1,1) = uy;  m(1,2) = uz;  m(1,3) = -(ux*eye[0] + uy*eye[1] + uz*eye[2]);
    m(2,0) = -fx; m(2,1) = -fy; m(2,2) = -fz; m(2,3) = fx*eye[0] + fy*eye[1] + fz*eye[2];
    m(3,3) = 1.0f;
    return m;
}

// OpenGL perspective: Z maps to [-1, 1].
static Mat4 perspective(float fov_y, float aspect, float near_z, float far_z) {
    float f = 1.0f / tanf(fov_y * 0.5f);
    Mat4 m = {};
    m(0,0) = f / aspect;
    m(1,1) = f;
    m(2,2) = (far_z + near_z) / (near_z - far_z);
    m(2,3) = 2.0f * far_z * near_z / (near_z - far_z);
    m(3,2) = -1.0f;
    return m;
}

// Build 4x4 model matrix from mjvGeom pos[3] + mat[9] + scale
static Mat4 geom_model_matrix(const mjvGeom &g, float sx, float sy, float sz) {
    Mat4 m = {};
    // MuJoCo g.mat is 3x3 row-major. OpenGL column-major: column j = (mat[j], mat[3+j], mat[6+j])
    m(0,0) = (float)g.mat[0]*sx; m(1,0) = (float)g.mat[3]*sx; m(2,0) = (float)g.mat[6]*sx;
    m(0,1) = (float)g.mat[1]*sy; m(1,1) = (float)g.mat[4]*sy; m(2,1) = (float)g.mat[7]*sy;
    m(0,2) = (float)g.mat[2]*sz; m(1,2) = (float)g.mat[5]*sz; m(2,2) = (float)g.mat[8]*sz;
    m(0,3) = (float)g.pos[0];    m(1,3) = (float)g.pos[1];    m(2,3) = (float)g.pos[2];
    m(3,3) = 1.0f;
    return m;
}

// Extract 3x3 normal matrix from model matrix (approximate for non-uniform scale)
static void extract_normal_mat3(const Mat4 &m, float out[9]) {
    // Column-major 3x3
    out[0] = m.m[0]; out[1] = m.m[1]; out[2] = m.m[2];
    out[3] = m.m[4]; out[4] = m.m[5]; out[5] = m.m[6];
    out[6] = m.m[8]; out[7] = m.m[9]; out[8] = m.m[10];
}

// ---------------------------------------------------------------------------
// Public API implementation
// ---------------------------------------------------------------------------

MujocoRenderer *mujoco_renderer_create(uint32_t width, uint32_t height) {
    MujocoRenderer *r = new MujocoRenderer();
    r->width = width;
    r->height = height;

    // Compile shaders
    GLuint vs = compile_shader(GL_VERTEX_SHADER, kVertexShaderSrc);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, kFragmentShaderSrc);
    if (!vs || !fs) { delete r; return nullptr; }
    r->prog_main = link_program(vs, fs);
    if (!r->prog_main) { delete r; return nullptr; }

    // Floor program (same vertex shader, different fragment)
    vs = compile_shader(GL_VERTEX_SHADER, kVertexShaderSrc);
    GLuint ffs = compile_shader(GL_FRAGMENT_SHADER, kFloorFragmentShaderSrc);
    r->prog_floor = link_program(vs, ffs);

    // Blit program
    GLuint bvs = compile_shader(GL_VERTEX_SHADER, kBlitVertexShaderSrc);
    GLuint bfs = compile_shader(GL_FRAGMENT_SHADER, kBlitFragmentShaderSrc);
    r->prog_blit = link_program(bvs, bfs);

    // Cache uniform locations — main
    r->loc_mvp        = glGetUniformLocation(r->prog_main, "uMVP");
    r->loc_model_mat  = glGetUniformLocation(r->prog_main, "uModelMat");
    r->loc_normal_mat = glGetUniformLocation(r->prog_main, "uNormalMat");
    r->loc_color      = glGetUniformLocation(r->prog_main, "uColor");
    r->loc_light_dir  = glGetUniformLocation(r->prog_main, "uLightDir");
    r->loc_light_dir2 = glGetUniformLocation(r->prog_main, "uLightDir2");
    r->loc_eye_pos    = glGetUniformLocation(r->prog_main, "uEyePos");
    r->loc_brightness = glGetUniformLocation(r->prog_main, "uBrightness");
    r->loc_contrast   = glGetUniformLocation(r->prog_main, "uContrast");

    // Cache uniform locations — floor
    if (r->prog_floor) {
        r->floc_mvp        = glGetUniformLocation(r->prog_floor, "uMVP");
        r->floc_model_mat  = glGetUniformLocation(r->prog_floor, "uModelMat");
        r->floc_normal_mat = glGetUniformLocation(r->prog_floor, "uNormalMat");
        r->floc_color      = glGetUniformLocation(r->prog_floor, "uColor");
        r->floc_light_dir  = glGetUniformLocation(r->prog_floor, "uLightDir");
        r->floc_light_dir2 = glGetUniformLocation(r->prog_floor, "uLightDir2");
        r->floc_eye_pos    = glGetUniformLocation(r->prog_floor, "uEyePos");
        r->floc_brightness = glGetUniformLocation(r->prog_floor, "uBrightness");
        r->floc_contrast   = glGetUniformLocation(r->prog_floor, "uContrast");
        r->floc_half_size  = glGetUniformLocation(r->prog_floor, "uFloorHalfSize");
        r->floc_grid_divs  = glGetUniformLocation(r->prog_floor, "uFloorGridDivs");
    }

    // Cache uniform locations — blit
    if (r->prog_blit) {
        r->bloc_uv_scale  = glGetUniformLocation(r->prog_blit, "uUVScale");
        r->bloc_uv_offset = glGetUniformLocation(r->prog_blit, "uUVOffset");
        r->bloc_texture   = glGetUniformLocation(r->prog_blit, "uTexture");
    }

    // Empty VAO for blit (uses gl_VertexID)
    glGenVertexArrays(1, &r->blit_vao);

    // Create FBO
    create_fbo(r);

    // Generate unit geometry
    int rings = 12, sectors = 16;
    r->sphere   = create_geom_buffer(make_unit_sphere(rings, sectors), make_sphere_indices(rings, sectors));
    r->capsule  = create_geom_buffer(make_unit_capsule(rings, sectors), make_capsule_indices(rings, sectors));
    r->cylinder = create_geom_buffer(make_unit_cylinder(sectors), make_cylinder_indices(sectors));
    r->box      = create_geom_buffer(make_unit_box(), make_box_indices());

    return r;
}

void mujoco_renderer_destroy(MujocoRenderer *r) {
    if (!r) return;

    destroy_geom_buffer(r->sphere);
    destroy_geom_buffer(r->capsule);
    destroy_geom_buffer(r->cylinder);
    destroy_geom_buffer(r->box);

    for (auto &[id, entry] : r->mesh_cache) {
        if (entry.vao) glDeleteVertexArrays(1, &entry.vao);
        if (entry.vbo) glDeleteBuffers(1, &entry.vbo);
    }
    r->mesh_cache.clear();

    if (r->skin_vao) glDeleteVertexArrays(1, &r->skin_vao);
    if (r->skin_vbo) glDeleteBuffers(1, &r->skin_vbo);
    if (r->skin_ebo) glDeleteBuffers(1, &r->skin_ebo);

    if (r->blit_vao) glDeleteVertexArrays(1, &r->blit_vao);

    destroy_fbo(r);

    if (r->prog_main)  glDeleteProgram(r->prog_main);
    if (r->prog_floor) glDeleteProgram(r->prog_floor);
    if (r->prog_blit)  glDeleteProgram(r->prog_blit);

    delete r;
}

void mujoco_renderer_resize(MujocoRenderer *r, uint32_t width, uint32_t height) {
    if (!r || (r->width == width && r->height == height)) return;
    r->width = width;
    r->height = height;
    destroy_fbo(r);
    create_fbo(r);
}

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
                            const float *arena_offset,
                            float brightness,
                            float contrast) {
    if (!r || !mj || !mj->loaded || !cam) return;

    // Save OpenGL state that ImGui expects
    GLint prev_fbo;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    GLint prev_viewport[4];
    glGetIntegerv(GL_VIEWPORT, prev_viewport);
    GLboolean prev_depth_test = glIsEnabled(GL_DEPTH_TEST);
    GLboolean prev_blend = glIsEnabled(GL_BLEND);
    GLboolean prev_cull = glIsEnabled(GL_CULL_FACE);

    // Update abstract scene
    mjv_updateScene(mj->model, mj->data, &mj->opt, nullptr, cam, mjCAT_ALL, &mj->scene);

    // Build view/projection matrices
    Mat4 vp;
    float eye[3];

    if (view_override && view_override->active) {
        Mat4 view, proj;
        memcpy(&view, view_override->view, sizeof(view));
        memcpy(&proj, view_override->proj, sizeof(proj));
        vp = mat4_mul(proj, view);
        eye[0] = view_override->eye[0];
        eye[1] = view_override->eye[1];
        eye[2] = view_override->eye[2];
    } else {
        float az = (float)cam->azimuth * (float)M_PI / 180.0f;
        float el = (float)cam->elevation * (float)M_PI / 180.0f;
        float ca_ = cosf(az), sa_ = sinf(az);
        float ce_ = cosf(el), se_ = sinf(el);
        float dist = (float)cam->distance;
        float center[3] = {(float)cam->lookat[0], (float)cam->lookat[1], (float)cam->lookat[2]};
        eye[0] = center[0] - dist * ce_ * ca_;
        eye[1] = center[1] - dist * ce_ * sa_;
        eye[2] = center[2] - dist * se_;
        float up[3] = {-se_ * ca_, -se_ * sa_, ce_};
        float aspect = (float)r->width / (float)r->height;
        Mat4 view = look_at(eye, center, up);
        Mat4 proj = perspective(45.0f * (float)M_PI / 180.0f, aspect, 0.001f, 100.0f);
        vp = mat4_mul(proj, view);
    }

    float light_dir[3], light_dir2[3];
    {
        float x = 0.2f, y = 0.1f, z = 1.0f;
        float len = sqrtf(x*x + y*y + z*z);
        light_dir[0] = x/len; light_dir[1] = y/len; light_dir[2] = z/len;
    }
    {
        float x = 0.5f, y = 0.8f, z = 0.3f;
        float len = sqrtf(x*x + y*y + z*z);
        light_dir2[0] = x/len; light_dir2[1] = y/len; light_dir2[2] = z/len;
    }

    // Bind FBO
    glBindFramebuffer(GL_FRAMEBUFFER, r->fbo);
    glViewport(0, 0, r->width, r->height);

    // Clear
    if (bg_texture) {
        glClearColor(0, 0, 0, 0);
    } else {
        glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw video background if provided
    if (bg_texture && r->prog_blit) {
        glDisable(GL_DEPTH_TEST);
        glUseProgram(r->prog_blit);
        glUniform2f(r->bloc_uv_scale, bg_zoom, bg_zoom);
        glUniform2f(r->bloc_uv_offset, bg_pan ? bg_pan[0] : 0.0f, bg_pan ? bg_pan[1] : 0.0f);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, (GLuint)(uintptr_t)bg_texture);
        glUniform1i(r->bloc_texture, 0);
        glBindVertexArray(r->blit_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }

    // Enable 3D rendering state
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // Use main program
    glUseProgram(r->prog_main);
    glUniform3fv(r->loc_light_dir, 1, light_dir);
    glUniform3fv(r->loc_light_dir2, 1, light_dir2);
    glUniform3fv(r->loc_eye_pos, 1, eye);
    glUniform1f(r->loc_brightness, brightness);
    glUniform1f(r->loc_contrast, contrast);

    // Helper: set per-geom uniforms and draw indexed geometry
    auto draw_geom = [&](const GeomBuffer &gb, const Mat4 &model_mat,
                          const mjvGeom &g, float scene_op) {
        Mat4 mvp = mat4_mul(vp, model_mat);
        float nm[9];
        extract_normal_mat3(model_mat, nm);
        float alpha = (g.objtype == mjOBJ_SITE) ? g.rgba[3] : 1.0f;
        float color[4] = {g.rgba[0], g.rgba[1], g.rgba[2], alpha * scene_op};

        glUniformMatrix4fv(r->loc_mvp, 1, GL_FALSE, mvp.m);
        glUniformMatrix4fv(r->loc_model_mat, 1, GL_FALSE, model_mat.m);
        glUniformMatrix3fv(r->loc_normal_mat, 1, GL_FALSE, nm);
        glUniform4fv(r->loc_color, 1, color);

        glBindVertexArray(gb.vao);
        glDrawElements(GL_TRIANGLES, gb.index_count, GL_UNSIGNED_INT, nullptr);
    };

    // Draw each geom
    for (int i = 0; i < mj->scene.ngeom; i++) {
        const mjvGeom &g = mj->scene.geoms[i];
        if (g.rgba[3] < 0.01f) continue;
        if (!show_sites && g.objtype == mjOBJ_SITE) continue;
        if (!show_bodies && g.objtype != mjOBJ_SITE) continue;

        switch (g.type) {
            case mjGEOM_SPHERE:
            case mjGEOM_ELLIPSOID: {
                float sx = g.size[0], sy = g.size[1], sz = g.size[2];
                if (g.type == mjGEOM_SPHERE) sy = sz = sx;
                draw_geom(r->sphere, geom_model_matrix(g, sx, sy, sz), g, scene_opacity);
                break;
            }
            case mjGEOM_CAPSULE: {
                float radius = g.size[0], half_len = g.size[2];
                draw_geom(r->capsule, geom_model_matrix(g, radius, radius, half_len), g, scene_opacity);
                break;
            }
            case mjGEOM_CYLINDER: {
                float radius = g.size[0], half_len = g.size[2];
                draw_geom(r->cylinder, geom_model_matrix(g, radius, radius, half_len), g, scene_opacity);
                break;
            }
            case mjGEOM_BOX: {
                draw_geom(r->box, geom_model_matrix(g, g.size[0], g.size[1], g.size[2]), g, scene_opacity);
                break;
            }
            case mjGEOM_MESH: {
                if (g.dataid < 0) continue;
                int meshid = g.dataid / 2;
                if (meshid >= mj->model->nmesh) continue;
                int nface = mj->model->mesh_facenum[meshid];
                if (nface == 0) continue;

                auto it = r->mesh_cache.find(meshid);
                if (it == r->mesh_cache.end()) {
                    int vertadr = mj->model->mesh_vertadr[meshid];
                    int normaladr = mj->model->mesh_normaladr[meshid];
                    int fadr = mj->model->mesh_faceadr[meshid];
                    int *fv = mj->model->mesh_face + fadr * 3;
                    int *fn = mj->model->mesh_facenormal + fadr * 3;
                    int ntri = nface;

                    std::vector<Vertex> verts(ntri * 3);
                    for (int f = 0; f < ntri; f++) {
                        for (int c = 0; c < 3; c++) {
                            int vi = fv[3*f + c] + vertadr;
                            int ni = fn[3*f + c] + normaladr;
                            verts[3*f + c] = {
                                mj->model->mesh_vert[3*vi],
                                mj->model->mesh_vert[3*vi+1],
                                mj->model->mesh_vert[3*vi+2],
                                mj->model->mesh_normal[3*ni],
                                mj->model->mesh_normal[3*ni+1],
                                mj->model->mesh_normal[3*ni+2]
                            };
                        }
                    }

                    MeshCacheEntry entry;
                    entry.tri_count = ntri;
                    glGenVertexArrays(1, &entry.vao);
                    glGenBuffers(1, &entry.vbo);
                    glBindVertexArray(entry.vao);
                    glBindBuffer(GL_ARRAY_BUFFER, entry.vbo);
                    glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_STATIC_DRAW);
                    glEnableVertexAttribArray(0);
                    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
                    glEnableVertexAttribArray(1);
                    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3*sizeof(float)));
                    glBindVertexArray(0);

                    r->mesh_cache[meshid] = entry;
                    it = r->mesh_cache.find(meshid);
                }

                int ntri = it->second.tri_count;
                Mat4 model_mat = geom_model_matrix(g, 1.0f, 1.0f, 1.0f);
                Mat4 mvp = mat4_mul(vp, model_mat);
                float nm[9];
                extract_normal_mat3(model_mat, nm);
                float a = (g.objtype == mjOBJ_SITE) ? g.rgba[3] : 1.0f;
                float color[4] = {g.rgba[0], g.rgba[1], g.rgba[2], a * scene_opacity};

                glUniformMatrix4fv(r->loc_mvp, 1, GL_FALSE, mvp.m);
                glUniformMatrix4fv(r->loc_model_mat, 1, GL_FALSE, model_mat.m);
                glUniformMatrix3fv(r->loc_normal_mat, 1, GL_FALSE, nm);
                glUniform4fv(r->loc_color, 1, color);

                glBindVertexArray(it->second.vao);
                glDrawArrays(GL_TRIANGLES, 0, ntri * 3);
                break;
            }
            default:
                continue;
        }
    }

    // --- Render skins ---
    for (int si = 0; show_skin && si < mj->scene.nskin; si++) {
        int nvert = mj->scene.skinvertnum[si];
        int nface = mj->scene.skinfacenum[si];
        if (nvert == 0 || nface == 0) continue;

        int vadr = mj->scene.skinvertadr[si];
        float *vpos = mj->scene.skinvert + vadr * 3;
        float *vnorm = mj->scene.skinnormal + vadr * 3;

        // Build interleaved vertex data
        std::vector<Vertex> verts(nvert);
        for (int v = 0; v < nvert; v++) {
            verts[v] = {vpos[3*v], vpos[3*v+1], vpos[3*v+2],
                        vnorm[3*v], vnorm[3*v+1], vnorm[3*v+2]};
        }

        if (!r->skin_vao) {
            glGenVertexArrays(1, &r->skin_vao);
            glGenBuffers(1, &r->skin_vbo);
            glGenBuffers(1, &r->skin_ebo);
        }

        glBindVertexArray(r->skin_vao);
        glBindBuffer(GL_ARRAY_BUFFER, r->skin_vbo);
        glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(Vertex), verts.data(), GL_DYNAMIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3*sizeof(float)));

        if (!r->skin_ebo_built) {
            int *faces = mj->model->skin_face + mj->model->skin_faceadr[si] * 3;
            std::vector<uint32_t> idx(nface * 3);
            for (int f = 0; f < nface * 3; f++) idx[f] = (uint32_t)faces[f];
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, r->skin_ebo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.size() * sizeof(uint32_t), idx.data(), GL_STATIC_DRAW);
            r->skin_face_count = nface;
            r->skin_ebo_built = true;
        }

        Mat4 identity = mat4_identity();
        float nm_id[9] = {1,0,0, 0,1,0, 0,0,1};
        float color[4] = {0.45f, 0.75f, 0.85f, 0.85f * scene_opacity};
        glUniformMatrix4fv(r->loc_mvp, 1, GL_FALSE, vp.m);
        glUniformMatrix4fv(r->loc_model_mat, 1, GL_FALSE, identity.m);
        glUniformMatrix3fv(r->loc_normal_mat, 1, GL_FALSE, nm_id);
        glUniform4fv(r->loc_color, 1, color);

        glDrawElements(GL_TRIANGLES, r->skin_face_count * 3, GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);
    }

    // --- Render arena (checkerboard floor) ---
    if (show_arena && r->prog_floor) {
        float hw = arena_width * 0.5f;
        float hd = arena_depth * 0.5f;
        float ox = arena_offset ? arena_offset[0] : 0.0f;
        float oy = arena_offset ? arena_offset[1] : 0.0f;
        float oz = arena_offset ? arena_offset[2] : 0.0f;

        Vertex floor_verts[4] = {
            {{ox-hw, oy-hd, oz}, {0, 0, 1}},
            {{ox+hw, oy-hd, oz}, {0, 0, 1}},
            {{ox+hw, oy+hd, oz}, {0, 0, 1}},
            {{ox-hw, oy+hd, oz}, {0, 0, 1}},
        };
        uint32_t floor_idx[6] = {0, 1, 2, 0, 2, 3};

        GLuint fvao, fvbo, febo;
        glGenVertexArrays(1, &fvao);
        glGenBuffers(1, &fvbo);
        glGenBuffers(1, &febo);
        glBindVertexArray(fvao);
        glBindBuffer(GL_ARRAY_BUFFER, fvbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(floor_verts), floor_verts, GL_STREAM_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, febo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(floor_idx), floor_idx, GL_STREAM_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(3*sizeof(float)));

        glUseProgram(r->prog_floor);
        Mat4 identity = mat4_identity();
        float nm_id[9] = {1,0,0, 0,1,0, 0,0,1};
        float color[4] = {0.25f, 0.32f, 0.42f, scene_opacity};
        glUniformMatrix4fv(r->floc_mvp, 1, GL_FALSE, vp.m);
        glUniformMatrix4fv(r->floc_model_mat, 1, GL_FALSE, identity.m);
        glUniformMatrix3fv(r->floc_normal_mat, 1, GL_FALSE, nm_id);
        glUniform4fv(r->floc_color, 1, color);
        glUniform3fv(r->floc_light_dir, 1, light_dir);
        glUniform3fv(r->floc_light_dir2, 1, light_dir2);
        glUniform3fv(r->floc_eye_pos, 1, eye);
        glUniform1f(r->floc_brightness, brightness);
        glUniform1f(r->floc_contrast, contrast);
        glUniform1f(r->floc_half_size, std::max(hw, hd));
        glUniform1i(r->floc_grid_divs, 4);

        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);

        glDeleteVertexArrays(1, &fvao);
        glDeleteBuffers(1, &fvbo);
        glDeleteBuffers(1, &febo);

        // Switch back to main program for corner markers
        glUseProgram(r->prog_main);
        glUniform3fv(r->loc_light_dir, 1, light_dir);
        glUniform3fv(r->loc_light_dir2, 1, light_dir2);
        glUniform3fv(r->loc_eye_pos, 1, eye);
        glUniform1f(r->loc_brightness, brightness);
        glUniform1f(r->loc_contrast, contrast);
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
        float corner_colors[4][4] = {
            {1.0f, 0.2f, 0.2f, 1.0f},
            {0.2f, 0.9f, 0.2f, 1.0f},
            {0.3f, 0.5f, 1.0f, 1.0f},
            {1.0f, 0.9f, 0.1f, 1.0f},
        };
        float corner_r = 0.025f;

        for (int ci = 0; ci < 4; ci++) {
            Mat4 model_mat = {};
            model_mat(0,0) = corner_r;
            model_mat(1,1) = corner_r;
            model_mat(2,2) = corner_r;
            model_mat(0,3) = corner_pos[ci][0];
            model_mat(1,3) = corner_pos[ci][1];
            model_mat(2,3) = corner_pos[ci][2];
            model_mat(3,3) = 1.0f;

            Mat4 mvp = mat4_mul(vp, model_mat);
            float nm[9];
            extract_normal_mat3(model_mat, nm);
            float color[4] = {corner_colors[ci][0], corner_colors[ci][1],
                               corner_colors[ci][2], corner_colors[ci][3] * scene_opacity};

            glUniformMatrix4fv(r->loc_mvp, 1, GL_FALSE, mvp.m);
            glUniformMatrix4fv(r->loc_model_mat, 1, GL_FALSE, model_mat.m);
            glUniformMatrix3fv(r->loc_normal_mat, 1, GL_FALSE, nm);
            glUniform4fv(r->loc_color, 1, color);

            glBindVertexArray(r->sphere.vao);
            glDrawElements(GL_TRIANGLES, r->sphere.index_count, GL_UNSIGNED_INT, nullptr);
        }
    }

    // Restore OpenGL state
    glBindVertexArray(0);
    glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo);
    glViewport(prev_viewport[0], prev_viewport[1], prev_viewport[2], prev_viewport[3]);
    if (!prev_depth_test) glDisable(GL_DEPTH_TEST); else glEnable(GL_DEPTH_TEST);
    if (!prev_blend) glDisable(GL_BLEND); else glEnable(GL_BLEND);
    if (!prev_cull) glDisable(GL_CULL_FACE); else glEnable(GL_CULL_FACE);
    glUseProgram(0);
}

ImTextureID mujoco_renderer_get_texture(MujocoRenderer *r) {
    if (!r || !r->color_tex) return (ImTextureID)0;
    return (ImTextureID)(intptr_t)r->color_tex;
}

void mujoco_renderer_get_size(MujocoRenderer *r, uint32_t *w, uint32_t *h) {
    if (r) { *w = r->width; *h = r->height; }
    else   { *w = 0; *h = 0; }
}

#endif // RED_HAS_MUJOCO
#endif // !__APPLE__
