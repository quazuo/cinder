#version 460

#extension GL_EXT_ray_tracing : require

layout (location = 0) rayPayloadInEXT vec3 hitValue;

layout(binding = 0) uniform samplerCube skyboxTexSampler;

void main() {
    hitValue = texture(skyboxTexSampler, gl_WorldRayDirectionEXT).rgb;
}