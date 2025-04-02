#version 450

#include "utils/bindless.glsl"

layout(location = 0) in vec3 localPosition;

layout(location = 0) out vec4 outColor;

layout (push_constant) uniform PushResourceIDs {
    uint ubo_id;
    uint equirectangular_map_id;
} constants;

vec2 sample_spherical_map(vec3 v) {
    const vec2 inv_atan = vec2(0.1591, 0.3183);

    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv *= inv_atan;
    uv += 0.5;

    return uv;
}

void main() {
    vec2 uv = sample_spherical_map(normalize(localPosition));
    vec3 color = texture(bindless_textures[constants.equirectangular_map_id], uv).rgb;
    outColor = vec4(color, 1.0);
}
