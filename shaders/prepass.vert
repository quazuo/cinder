#version 450

#include "utils/bindless.glsl"
#include "utils/ubo.glsl"

layout (location = 0) in vec3 inPosition;
layout (location = 1) in vec2 inTexCoord;
layout (location = 2) in vec3 inNormal;
layout (location = 3) in vec3 inTangent;
layout (location = 4) in vec3 inBitangent;
layout (location = 5) in mat4 inInstanceTransform;

layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out vec3 fragPos;
layout (location = 2) out vec3 normal;

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
    uint ubo_id = constants.ubo_id;

    const mat4 model = ubos[ubo_id].matrices.model * inInstanceTransform;

    vec4 view_pos = ubos[ubo_id].matrices.view * model * vec4(inPosition, 1.0);
    fragPos = view_pos.xyz;
    gl_Position = ubos[ubo_id].matrices.proj * view_pos;

    fragTexCoord = inTexCoord;

    mat3 normal_matrix = transpose(inverse(mat3(ubos[ubo_id].matrices.view * model)));
    normal = normal_matrix * inNormal;
}
