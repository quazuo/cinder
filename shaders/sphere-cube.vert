#version 450

#extension GL_EXT_multiview : enable

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec3 localPosition;

layout (push_constant) uniform PushResourceIDs {
    uint ubo_id;
    uint equirectangular_map_id;
} constants;

layout(set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubos[];

void main() {
    localPosition = inPosition;

    const mat4 view = ubos[constants.ubo_id].matrices.cubemap_capture_views[gl_ViewIndex];
    gl_Position = ubos[constants.ubo_id].matrices.cubemap_capture_proj * view * vec4(inPosition, 1.0);
}
