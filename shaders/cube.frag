#version 450

#include "utils/bindless.glsl"

layout (location = 0) out vec4 outColor;

layout (push_constant) uniform PushResourceIDs {
    uint general_ubo_id;
} constants;

void main() {
    outColor = vec4(1, 0, 0, 1);
}
