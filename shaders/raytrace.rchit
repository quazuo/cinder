#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_buffer_reference2 : require

#include "utils/ubo.glsl"

layout (location = 0) rayPayloadInEXT vec3 hitValue;

layout (set = 0, binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;
layout (set = 0, binding = 1) uniform accelerationStructureEXT topLevelAS;

struct Vertex {
    vec3 pos;
    vec2 tex_coord;
    vec3 normal;
    vec3 tangent;
    vec3 bitangent;
};

#define MATERIAL_TEX_ARRAY_SIZE 32

layout (set = 1, binding = 0) uniform sampler2D baseColorSamplers[MATERIAL_TEX_ARRAY_SIZE];
layout (set = 1, binding = 1) uniform sampler2D normalSamplers[MATERIAL_TEX_ARRAY_SIZE];
layout (set = 1, binding = 2) uniform sampler2D ormSamplers[MATERIAL_TEX_ARRAY_SIZE];

struct Mesh {
    uint material_id;
    uint vertex_offset;
    uint index_offset;
};

layout (set = 2, binding = 0) readonly buffer Meshes {
    Mesh meshes[];
};
layout (set = 2, binding = 1) readonly buffer Vertices {
    Vertex vertices[];
};
layout (set = 2, binding = 2) readonly buffer Indices {
    uint indices[];
};

hitAttributeEXT vec3 attribs;

void main() {
    Mesh mesh = meshes[gl_InstanceCustomIndexEXT];

    uvec3 tri_indices = uvec3(
        indices[mesh.index_offset + 3 * gl_PrimitiveID],
        indices[mesh.index_offset + 3 * gl_PrimitiveID + 1],
        indices[mesh.index_offset + 3 * gl_PrimitiveID + 2]
    );

    Vertex v0 = vertices[mesh.vertex_offset + tri_indices.x];
    Vertex v1 = vertices[mesh.vertex_offset + tri_indices.y];
    Vertex v2 = vertices[mesh.vertex_offset + tri_indices.z];

    vec3 barycentrics = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);

    vec3 pos = v0.pos * barycentrics.x + v1.pos * barycentrics.y + v2.pos * barycentrics.z;
    vec3 world_pos = vec3(gl_ObjectToWorldEXT * vec4(pos, 1.0)); // transform the position to world space

    vec3 normal = v0.normal * barycentrics.x + v1.normal * barycentrics.y + v2.normal * barycentrics.z;
    vec3 world_normal = normalize(vec3(normal * gl_WorldToObjectEXT)); // transform the normal to world space

    vec2 tex_coord = v0.tex_coord * barycentrics.x + v1.tex_coord * barycentrics.y + v2.tex_coord * barycentrics.z;

    vec4 base_color = texture(baseColorSamplers[mesh.material_id], tex_coord);

    vec3 color = base_color.rgb;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    hitValue = color;
}
