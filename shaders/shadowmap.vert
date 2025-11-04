#version 450

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 5) in mat4 inInstanceTransform;

layout(set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform GeneralUniforms {
    WindowRes window;
    Matrices matrices;
    LightData light;
    MiscData misc;
} ubos[];

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
} constants;

void main() {
    const uint ubo_id = constants.general_ubo_id;

    const mat4 model = ubos[ubo_id].matrices.model * inInstanceTransform;
    const mat4 mvp = ubos[ubo_id].light.proj_x_view * model;
    // mvp = ubos[ubo_id].matrices.proj * ubos[ubo_id].matrices.view * model;
    gl_Position = mvp * vec4(inPosition, 1.0);
}
