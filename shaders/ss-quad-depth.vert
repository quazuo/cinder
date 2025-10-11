#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_tex_coords;

layout(location = 0) out vec2 out_tex_coords;

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
    uint sampled_tex_id;
} constants;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);

    out_tex_coords = in_tex_coords;
}
