#extension GL_EXT_nonuniform_qualifier : enable

#define BINDLESS_SET 0

#define BINDLESS_SAMPLER_BINDING 0
#define BINDLESS_STORAGE_TEXTURE_BINDING 1
#define BINDLESS_UBO_BINDING 2

layout(set = BINDLESS_SET, binding = BINDLESS_SAMPLER_BINDING) uniform sampler2D   bindless_samplers[];
layout(set = BINDLESS_SET, binding = BINDLESS_SAMPLER_BINDING) uniform sampler3D   bindless_samplers_3d[];
layout(set = BINDLESS_SET, binding = BINDLESS_SAMPLER_BINDING) uniform samplerCube bindless_samplers_cube[];

layout(rgba8, set = BINDLESS_SET, binding = BINDLESS_STORAGE_TEXTURE_BINDING) uniform image2D   bindless_textures[];
layout(rgba8, set = BINDLESS_SET, binding = BINDLESS_STORAGE_TEXTURE_BINDING) uniform image3D   bindless_textures_3d[];
layout(rgba8, set = BINDLESS_SET, binding = BINDLESS_STORAGE_TEXTURE_BINDING) uniform imageCube bindless_textures_cube[];
