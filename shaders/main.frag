#version 450

#extension GL_ARB_separate_shader_objects : enable

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"
#include "utils/material.glsl"

layout (location = 0) in vec3 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in vec4 lightSpacePosition;
layout (location = 3) in mat3 TBN;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
    uint ssao_tex_id;
    uint shadowmap_id;
    uint material_ubo_id;
    uint material_id;
} constants;

layout(set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform GeneralUniforms {
    WindowRes window;
    Matrices matrices;
    LightData light;
    MiscData misc;
} ubos[];

layout(set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform MaterialUniforms {
    Material mats[MATERIAL_MAX_COUNT];
} materials[];

vec4 sample_texture_with_fallback(uint tex_id, vec2 tex_coord) {
    if (tex_id == 0xffffffff) {
        return vec4(1, 1, 1, 1);
    }

    return texture(bindless_samplers[tex_id], tex_coord);
}

float get_blurred_ssao() {
    uint ubo_id = constants.general_ubo_id;
    vec2 texCoord = gl_FragCoord.xy / vec2(ubos[ubo_id].window.width, ubos[ubo_id].window.height);

    vec2 texelSize = vec2(1.0) / vec2(textureSize(bindless_samplers[constants.ssao_tex_id], 0));
    float result = 0.0;

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            result += sample_texture_with_fallback(constants.ssao_tex_id, texCoord + offset).r;
        }
    }

    return result / (4.0 * 4.0);
}

float calc_shadow() {
    vec3 proj_coords = lightSpacePosition.xyz / lightSpacePosition.w;
    // proj_coords = proj_coords * 0.5 + 0.5;

    float closest_depth = sample_texture_with_fallback(constants.shadowmap_id, proj_coords.xy).r;
    float current_depth = proj_coords.z;
    float shadow = current_depth > closest_depth ? 1.0 : 0.0;

    return shadow;
}

void main() {
    const uint ubo_id           = constants.general_ubo_id;
    const uint mat_ubo_id       = constants.material_ubo_id;
    const uint base_color_id    = materials[mat_ubo_id].mats[constants.material_id].base_color;
    const uint normal_id        = materials[mat_ubo_id].mats[constants.material_id].normal;
    const uint orm_id           = materials[mat_ubo_id].mats[constants.material_id].orm;

    vec4 base_color = sample_texture_with_fallback(base_color_id, fragTexCoord);

    if (base_color.a < 0.1) discard;

    vec3 normal = sample_texture_with_fallback(normal_id, fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);

    vec3 orm = sample_texture_with_fallback(orm_id, fragTexCoord).rgb;
    float ao = ubos[ubo_id].misc.use_ssao == 1u ? get_blurred_ssao() : orm.r;
    float roughness = orm.g;
    float metallic = orm.b;

    // light related values
    vec3 light_dir = normalize(ubos[ubo_id].light.direction);
    vec3 light_color = ubos[ubo_id].light.intensity * ubos[ubo_id].light.color;

    // utility vectors
    vec3 view = normalize(ubos[ubo_id].misc.camera_pos - worldPosition);
    vec3 halfway = normalize(view + light_dir);

    vec3 ambient = /* vec3(0.03) * */ base_color.rgb * ao;

    vec3 color = ambient;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    color = vec3(calc_shadow());

    outColor = vec4(color, 1.0);
}