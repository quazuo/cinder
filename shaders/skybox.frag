#version 450

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout(location = 0) in vec3 texCoord;

layout(location = 0) out vec4 outColor;

layout (push_constant) uniform PushResourceIDs {
    uint ubo_id;
    uint skybox_tex_id;
} constants;

void main() {
    vec3 color = texture(bindless_samplers_cube[constants.skybox_tex_id], texCoord).rgb;

    color = pow(color, vec3(1 / 2.2));

    outColor = vec4(color, 1.0);
}
