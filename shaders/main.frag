#version 450

#extension GL_ARB_separate_shader_objects : enable

#include "utils/ubo.glsl"
#include "utils/bindless.glsl"

layout (location = 0) in vec3 worldPosition;
layout (location = 1) in vec2 fragTexCoord;
layout (location = 2) in mat3 TBN;

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushConstants {
    uint material_id;
} constants;

layout (set = BINDLESS_DESCRIPTOR_SET, binding = BINDLESS_UNIFORM_BUFFER_BINDING) \
uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubo[];

layout (set = BINDLESS_DESCRIPTOR_SET, binding = BINDLESS_TEXTURE_SAMPLER_BINDING) \
uniform sampler2D globalTextures2d[];

layout (set = FRAGMENT_BINDLESS_PARAM_SET, binding = 0) uniform Params {
    uint uniformBuffer;
    uint gBufferNormal;
    uint gBufferPos;
    uint ssaoSampler;
} params;

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
    vec4 base_color = texture(globalTextures2d[...], fragTexCoord);

    if (base_color.a < 0.1) discard;

    vec3 normal = texture(globalTextures2d[...], fragTexCoord).rgb;
    normal = normalize(normal * 2.0 - 1.0);
    normal = normalize(TBN * normal);

    vec3 orm = texture(globalTextures2d[...], fragTexCoord).rgb;
    float ao = ubo.misc.use_ssao == 1u ? getBlurredSsao() : orm.r;
    float roughness = orm.g;
    float metallic = orm.b;

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
