#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTexCoords;

layout(location = 0) out vec2 outTexCoords;

layout (push_constant) uniform PushResourceIDs {
    uint ubo_id;
    uint g_depth_tex_id;
    uint g_normal_tex_id;
    uint g_pos_tex_id;
} constants;

void main() {
    gl_Position = vec4(inPosition, 0.0, 1.0);

    outTexCoords = inTexCoords;
}
