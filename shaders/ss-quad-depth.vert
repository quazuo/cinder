#version 450

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_tex_coords;

layout(location = 0) out vec2 out_tex_coords;

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
    uint sampled_tex_id;

    float bottom;
    float top;
    float left;
    float right;

    // used in the fragment shader
    float layer;
} constants;

void main() {
    vec2 out_position = in_position; // [-1, 1]
    out_position = out_position * 0.5f + 0.5f; // [0, 1]

    out_position.x *= constants.top - constants.bottom; // [0, t - b]
    out_position.x += constants.bottom; // [b, t]

    out_position.y *= constants.right - constants.left; // [0, r - l]
    out_position.y += constants.left; // [l, r]

    gl_Position = vec4(out_position, 0.0, 1.0);

    out_tex_coords = in_tex_coords;
}
