#version 450

#include "utils/bindless.glsl"

layout(location = 0) in vec2 tex_coords;

layout(location = 0) out vec4 out_color;

layout (push_constant) uniform PushResourceIDs {
    uint sampled_tex_id;
} constants;

void main() {
    out_color = vec4(texture(bindless_samplers[constants.sampled_tex_id], tex_coords).rgb, 1.0);
}
