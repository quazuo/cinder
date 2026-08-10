#version 450

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout(location = 0) in vec2 tex_coords;

layout(location = 0) out vec4 out_color;

layout (set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    LightData light;
    MiscData misc;
} ubos[];

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
    uint sampled_tex_id;

    // these are used in the vertex shader, not here
    float bottom;
    float top;
    float left;
    float right;

    float layer;
} constants;

float linearize_depth(float d) {
    float z_near = ubos[constants.general_ubo_id].misc.z_near;
    float z_far  = ubos[constants.general_ubo_id].misc.z_far;

    return z_near * z_far / (z_far + d * (z_near - z_far));
}

void main() {
    float depth = texture(bindless_samplers_2d_array[constants.sampled_tex_id], vec3(tex_coords, float(constants.layer))).r;
    float linearized = linearize_depth(depth);
    out_color = vec4(depth); // vec4(depth == 1.0 ? 1.0 : 0.0);
}
