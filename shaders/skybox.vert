#version 450

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout (location = 0) in vec3 inPosition;

layout (location = 0) out vec3 texCoord;

layout (push_constant) uniform PushResourceIDs {
    uint ubo_id;
    uint skybox_tex_id;
} constants;

layout (set = BINDLESS_SET, binding = BINDLESS_UBO_BINDING) uniform UniformBufferObject {
    WindowRes window;
    Matrices matrices;
    MiscData misc;
} ubos[];

void main() {
    texCoord = inPosition;

    const vec4 pos = ubos[constants.ubo_id].matrices.proj
                     * ubos[constants.ubo_id].matrices.static_view
                     * vec4(inPosition, 1.0);
    gl_Position = pos.xyww;
}
