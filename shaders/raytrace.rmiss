#version 460

#extension GL_EXT_ray_tracing : require

layout (location = 0) rayPayloadInEXT vec3 hitValue;

layout(set = 1, binding = 3) uniform samplerCube skyboxTexSampler;

void main() {
    vec3 color = texture(skyboxTexSampler, gl_WorldRayDirectionEXT).rgb;

    // apply hdr tonemapping
    color = color / (color + vec3(1.0));

    // apply gamma correction
    color = pow(color, vec3(1 / 2.2));

    hitValue = color;
}
