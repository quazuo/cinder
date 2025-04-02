#version 450

#extension GL_ARB_separate_shader_objects : enable

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout (location = 0) in vec3 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in mat3 TBN;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushResourceIDs {
    uint ubo_id;
    uint ssao_tex_id;
    uint base_color_tex_id;
    uint normal_tex_id;
    uint orm_tex_id;
} constants;

#define material_id 0

layout (set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubos[];

//layout (set = 0, binding = 1) uniform sampler2D ssaoSampler;
//
//// 32
//#define MATERIAL_TEX_ARRAY_SIZE 1
//
//layout (set = 1, binding = 0) uniform sampler2D baseColorSamplers[MATERIAL_TEX_ARRAY_SIZE];
//layout (set = 1, binding = 1) uniform sampler2D normalSamplers[MATERIAL_TEX_ARRAY_SIZE];
//layout (set = 1, binding = 2) uniform sampler2D ormSamplers[MATERIAL_TEX_ARRAY_SIZE];

float getBlurredSsao() {
    uint ubo_id = constants.ubo_id;
    vec2 texCoord = gl_FragCoord.xy / vec2(ubos[ubo_id].window.width, ubos[ubo_id].window.height);

    vec2 texelSize = vec2(1.0) / vec2(textureSize(bindless_textures[constants.ssao_tex_id], 0));
    float result = 0.0;

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            result += texture(bindless_textures[constants.ssao_tex_id], texCoord + offset).r;
        }
    }

    return result / (4.0 * 4.0);
}

void main() {
    const uint ubo_id = constants.ubo_id;

    vec4 base_color = texture(bindless_textures[constants.base_color_tex_id], fragTexCoord);

    if (base_color.a < 0.1) discard;

    vec3 normal = texture(bindless_textures[constants.normal_tex_id], fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);

    vec3 orm = texture(bindless_textures[constants.orm_tex_id], fragTexCoord).rgb;
    float ao = ubos[ubo_id].misc.use_ssao == 1u ? getBlurredSsao() : orm.r;
    float roughness = orm.g;
    float metallic = orm.b;

    // light related values
    vec3 light_dir = normalize(ubos[ubo_id].misc.light_direction);
    vec3 light_color = ubos[ubo_id].misc.light_color;

    // utility vectors
    vec3 view = normalize(ubos[ubo_id].misc.camera_pos - worldPosition);
    vec3 halfway = normalize(view + light_dir);

    vec3 ambient = vec3(0.03) * base_color.rgb * ao;

    vec3 color = ambient;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    outColor = vec4(color, 1.0);
}