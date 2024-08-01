#version 450

#extension GL_ARB_separate_shader_objects : enable

#include "utils/ubo.glsl"

layout (location = 0) in vec3 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in mat3 TBN;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushConstants {
    uint material_id;
} constants;

layout (set = 0, binding = 0) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo;

layout (set = 0, binding = 1) uniform sampler2D ssaoSampler;

#define MATERIAL_TEX_ARRAY_SIZE 32

layout (set = 1, binding = 0) uniform sampler2D baseColorSamplers[MATERIAL_TEX_ARRAY_SIZE];
layout (set = 1, binding = 1) uniform sampler2D normalSamplers[MATERIAL_TEX_ARRAY_SIZE];
layout (set = 1, binding = 2) uniform sampler2D ormSamplers[MATERIAL_TEX_ARRAY_SIZE];

float getBlurredSsao() {
    vec2 texCoord = gl_FragCoord.xy / vec2(ubo.window.width, ubo.window.height);

    vec2 texelSize = vec2(1.0) / vec2(textureSize(ssaoSampler, 0));
    float result = 0.0;

    for (int x = -2; x < 2; x++) {
        for (int y = -2; y < 2; y++) {
            vec2 offset = vec2(x, y) * texelSize;
            result += texture(ssaoSampler, texCoord + offset).r;
        }
    }

    return result / (4.0 * 4.0);
}

void main() {
    vec4 base_color = texture(baseColorSamplers[constants.material_id], fragTexCoord);

    if (base_color.a < 0.1) discard;

    vec3 normal = texture(normalSamplers[constants.material_id], fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);

    float ao = ubo.misc.use_ssao == 1u
        ? getBlurredSsao()
        : texture(ormSamplers[constants.material_id], fragTexCoord).r;
    float roughness = texture(ormSamplers[constants.material_id], fragTexCoord).g;
    float metallic = texture(ormSamplers[constants.material_id], fragTexCoord).b;

    // light related values
    vec3 light_dir = normalize(ubo.misc.light_direction);
    vec3 light_color = ubo.misc.light_color;

    // utility vectors
    vec3 view = normalize(ubo.misc.camera_pos - worldPosition);
    vec3 halfway = normalize(view + light_dir);

    vec3 ambient = vec3(0.03) * base_color.rgb * ao;

    vec3 color = ambient;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    outColor = vec4(color, 1.0);
}
