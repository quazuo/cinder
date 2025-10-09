#version 450

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout(location = 0) in vec3 world_position;

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
    gl_Position = ubos[constants.general_ubo_id].light.proj_x_view * vec4(world_position, 1.0);
}
