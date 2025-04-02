#version 450

#include "utils/bindless.glsl"

layout (location = 0) in vec2 texCoord;
layout (location = 1) in vec3 fragPos;
layout (location = 2) in vec3 normal;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec3 outPos;

void main() {
    outNormal = normalize(normal);

    outPos = fragPos;
}
