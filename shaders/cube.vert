#version 450

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout (location = 0) in vec3 inPosition;

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    vec4 box_min;
    vec4 box_max;
} constants;

layout (set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    LightData light;
    MiscData misc;
} ubos[];

void main() {
    const uint ubo_id = constants.general_ubo_id;

    vec3 out_position = vec3(0);

    if (inPosition.x == -1.0f) out_position.x = constants.box_min.x;
    if (inPosition.x ==  1.0f) out_position.x = constants.box_max.x;

    if (inPosition.y == -1.0f) out_position.y = constants.box_min.y;
    if (inPosition.y ==  1.0f) out_position.y = constants.box_max.y;

    if (inPosition.z == -1.0f) out_position.z = constants.box_min.z;
    if (inPosition.z ==  1.0f) out_position.z = constants.box_max.z;

    const mat4 mvp = ubos[ubo_id].matrices.proj * ubos[ubo_id].matrices.view;
    gl_Position = mvp * vec4(out_position, 1.0);
}