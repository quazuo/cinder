#extension GL_EXT_nonuniform_qualifier : enable

#define BINDLESS_SET 0
#define BINDLESS_TEXTURE_BINDING 0
#define BINDLESS_UBO_BINDING 1

layout(set = BINDLESS_SET, binding = BINDLESS_TEXTURE_BINDING) uniform sampler2D bindless_textures[];
layout(set = BINDLESS_SET, binding = BINDLESS_TEXTURE_BINDING) uniform sampler3D bindless_textures_3d[];
layout(set = BINDLESS_SET, binding = BINDLESS_TEXTURE_BINDING) uniform samplerCube bindless_textures_cube[];
