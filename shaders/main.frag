#version 450

#extension GL_ARB_separate_shader_objects : enable

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"
#include "utils/model.glsl"

layout (location = 0) in vec4 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in mat3 TBN;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
    uint mesh_desc_ubo_id;
    uint ssao_tex_id;
    uint shadowmap_id;

    uint mesh_id;
} constants;

layout(set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform GeneralUniforms {
    WindowRes window;
    Matrices matrices;
    LightData light;
    MiscData misc;
} ubos[];

layout(set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform MeshDescriptionUniforms {
    MeshDescription md[MAX_MODEL_MESH_COUNT];
} mesh_descriptions[];

vec4 sample_texture_with_fallback(uint tex_id, vec2 tex_coord) {
    if (tex_id == 0xffffffff) {
        return vec4(1, 1, 1, 1);
    }

    return texture(bindless_samplers[tex_id], tex_coord);
}

vec4 sample_texture_layer_with_fallback(uint tex_id, vec2 tex_coord, uint layer) {
    if (tex_id == 0xffffffff) {
        return vec4(1, 1, 1, 1);
    }

    return textureLod(bindless_samplers_2d_array[tex_id], vec3(tex_coord, float(layer)), 0);
}

vec3 get_normal() {
    const MeshDescription mesh_desc = mesh_descriptions[constants.mesh_desc_ubo_id].md[constants.mesh_id];
    vec3 normal = sample_texture_with_fallback(mesh_desc.normal_id, fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);
    return normal;
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

float calc_shadow(vec2 texel_offset, uint layer) {
    const uint ubo_id = constants.general_ubo_id;

    vec2 shadowmap_texel_size = vec2(1.0f / 2048.0f); // textureSize(bindless_samplers[constants.shadowmap_id], 0).xy;
    vec2 tex_coord_offset = texel_offset * shadowmap_texel_size;

    vec4 light_space_pos = ubos[ubo_id].light.cascade_pxv_mats[layer] * worldPosition;
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords.xy = proj_coords.xy * 0.5 + 0.5;
    proj_coords.y = 1.0f - proj_coords.y;

    vec3 normal = get_normal();
    vec3 light_direction = ubos[ubo_id].light.direction;
    float bias_weight_1  = ubos[ubo_id].misc.bias_weight_1;
    float bias_weight_2  = ubos[ubo_id].misc.bias_weight_2;
    float texel_world_size = ubos[ubo_id].light.cascade_infos[layer].texel_world_size;
    float bias = max(
        bias_weight_1 * texel_world_size * (1.0 - dot(normal, light_direction)),
        bias_weight_2 * texel_world_size
    );

    float closest_depth = sample_texture_layer_with_fallback(constants.shadowmap_id, proj_coords.xy + tex_coord_offset, layer).r;
    float current_depth = proj_coords.z;
    float shadow = current_depth - bias > closest_depth ? 1.0 : 0.0;

    return shadow;
}

float calc_shadow_with_pcf(uint layer) {
    const int pcf_radius = 1;
    float shadow_amount = 0.0f;

    for (int off_x = -pcf_radius; off_x <= pcf_radius; off_x++) {
        for (int off_y = -pcf_radius; off_y <= pcf_radius; off_y++) {
            if (calc_shadow(vec2(off_x, off_y), layer) == 1.0f) {
                shadow_amount += 1.0f;
            }
        }
    }

    shadow_amount /= (2.0f * pcf_radius + 1.0f) * (2.0f * pcf_radius + 1.0f);
    return shadow_amount;
}

void main() {
    const MeshDescription mesh_desc = mesh_descriptions[constants.mesh_desc_ubo_id].md[constants.mesh_id];
    const uint ubo_id = constants.general_ubo_id;

    vec4 base_color = sample_texture_with_fallback(mesh_desc.base_color_id, fragTexCoord);
    if (base_color.a < 0.1) discard;

    vec3 normal = sample_texture_with_fallback(mesh_desc.normal_id, fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);

    vec3 orm = sample_texture_with_fallback(mesh_desc.orm_id, fragTexCoord).rgb;
    float ao = ubos[ubo_id].misc.use_ssao == 1u ? get_blurred_ssao() : orm.r;
    float roughness = orm.g;
    float metallic = orm.b;

    // light related values
    vec3 light_dir = normalize(ubos[ubo_id].light.direction);
    vec3 light_color = ubos[ubo_id].light.intensity * ubos[ubo_id].light.color;

    // utility vectors
    vec3 view = normalize(ubos[ubo_id].misc.camera_pos - worldPosition.xyz);
    vec3 halfway = normalize(view + light_dir);

    vec3 ambient = /* vec3(0.03) * */ base_color.rgb * ao;

    vec3 color = ambient;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    // select CSM cascade
    const uint CASCADE_COUNT = 4;
    vec4 frag_pos_view_space = ubos[ubo_id].matrices.view * vec4(worldPosition.xyz, 1.0);
    float depth = abs(frag_pos_view_space.z);
    uint layer = 0;
    float lerp_alpha_with_lower = 0.0f;
    float lerp_alpha_with_upper = 0.0f;

    for (uint i = 0; i < CASCADE_COUNT; i++) {
        const float lower_cascade_z_far = ubos[ubo_id].light.cascade_infos[i - 1].z_far;
        const float cascade_z_far = ubos[ubo_id].light.cascade_infos[i].z_far;
        const float blend_threshold = ubos[ubo_id].light.cascade_blend_threshold;

        if (depth < cascade_z_far) {
            layer = i;

            const float lower_blend_threshold = lower_cascade_z_far + cascade_z_far * blend_threshold * 0.3f;
            const float upper_blend_threshold = cascade_z_far * (1.0f - blend_threshold);

            if (layer > 0 && depth < lower_blend_threshold) {
                lerp_alpha_with_lower = 0.5f * (depth - lower_blend_threshold) / (lower_cascade_z_far - lower_blend_threshold);

            } else if (layer < CASCADE_COUNT - 1 && depth > upper_blend_threshold) {
                lerp_alpha_with_upper = 0.5f * (depth - upper_blend_threshold) / (cascade_z_far - upper_blend_threshold);
            }

            break;
        }
    }

    // apply shadows
    const float shadow_factor = 0.3f;

    float pcf_shadow_amount = calc_shadow_with_pcf(layer);

    if (lerp_alpha_with_lower != 0.0f) {
        float lower_layer_shadow_amount = calc_shadow_with_pcf(layer - 1);
        pcf_shadow_amount = mix(pcf_shadow_amount, lower_layer_shadow_amount, lerp_alpha_with_lower);

    } else if (lerp_alpha_with_upper != 0.0f) {
        float upper_layer_shadow_amount = calc_shadow_with_pcf(layer + 1);
        pcf_shadow_amount = mix(pcf_shadow_amount, upper_layer_shadow_amount, lerp_alpha_with_upper);
    }

    color = mix(color, color * shadow_factor, pcf_shadow_amount);

    outColor = vec4(color, 1.0);
}