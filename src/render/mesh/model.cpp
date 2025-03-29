#include "model.hpp"

#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "vertex.hpp"
#include "src/render/renderer.hpp"
#include "src/render/vk/image.hpp"
#include "src/render/vk/buffer.hpp"

namespace zrx {
static glm::vec3 assimp_vec_to_glm(const aiVector3D &v) {
    return {v.x, v.y, v.z};
}

static glm::mat4 assimp_matrix_to_glm(const aiMatrix4x4 &m) {
    glm::mat4 res;

    res[0][0] = m.a1;
    res[1][0] = m.a2;
    res[2][0] = m.a3;
    res[3][0] = m.a4;

    res[0][1] = m.b1;
    res[1][1] = m.b2;
    res[2][1] = m.b3;
    res[3][1] = m.b4;

    res[0][2] = m.c1;
    res[1][2] = m.c2;
    res[2][2] = m.c3;
    res[3][2] = m.c4;

    res[0][3] = m.d1;
    res[1][3] = m.d2;
    res[2][3] = m.d3;
    res[3][3] = m.d4;

    return res;
}

Mesh::Mesh(const aiMesh *assimp_mesh) : material_id(assimp_mesh->mMaterialIndex) {
    std::unordered_map<ModelVertex, uint32_t> unique_vertices;

    for (size_t faceIdx = 0; faceIdx < assimp_mesh->mNumFaces; faceIdx++) {
        const auto &face = assimp_mesh->mFaces[faceIdx];

        for (size_t i = 0; i < face.mNumIndices; i++) {
            ModelVertex vertex{};

            if (assimp_mesh->HasPositions()) {
                vertex.pos = assimp_vec_to_glm(assimp_mesh->mVertices[face.mIndices[i]]);
            }

            if (assimp_mesh->HasTextureCoords(0)) {
                vertex.tex_coord = {
                    assimp_mesh->mTextureCoords[0][face.mIndices[i]].x,
                    1.0f - assimp_mesh->mTextureCoords[0][face.mIndices[i]].y
                };
            }

            if (assimp_mesh->HasTangentsAndBitangents()) {
                vertex.normal = assimp_vec_to_glm(assimp_mesh->mNormals[face.mIndices[i]]);
            }

            if (assimp_mesh->HasTangentsAndBitangents()) {
                vertex.tangent   = assimp_vec_to_glm(assimp_mesh->mTangents[face.mIndices[i]]);
                vertex.bitangent = assimp_vec_to_glm(assimp_mesh->mBitangents[face.mIndices[i]]);
            }

            if (!unique_vertices.contains(vertex)) {
                unique_vertices[vertex] = vertices.size();
                vertices.push_back(vertex);
            }

            indices.push_back(unique_vertices.at(vertex));
        }
    }
}

Material::Material(const RendererContext &ctx, const aiMaterial *assimp_material,
                   const std::filesystem::path &base_path) {
    // base color

    aiString base_color_rel_path;
    aiReturn result = assimp_material->GetTexture(aiTextureType_BASE_COLOR, 0, &base_color_rel_path);

    if (result == aiReturn_SUCCESS) {
        auto path = base_path;
        path /= base_color_rel_path.C_Str();
        path.make_preferred();

        try {
            base_color = TextureBuilder()
                    .with_flags(vk::TextureFlagBitsZRX::MIPMAPS)
                    .from_paths({path})
                    .create(ctx);
        } catch (std::exception &e) {
            std::cerr << "failed to allocate buffer for texture: " << path << std::endl;
            base_color = nullptr;
        }
    }

    // normal map

    aiString normal_rel_path;
    if (assimp_material->GetTexture(aiTextureType_NORMALS, 0, &normal_rel_path) != aiReturn_SUCCESS) {
        result = assimp_material->GetTexture(aiTextureType_NORMAL_CAMERA, 0, &normal_rel_path);
    }

    if (result == aiReturn_SUCCESS) {
        auto path = base_path;
        path /= normal_rel_path.C_Str();
        path.make_preferred();

        normal = TextureBuilder()
                .use_format(vk::Format::eR8G8B8A8Unorm)
                .from_paths({path})
                .with_flags(vk::TextureFlagBitsZRX::MIPMAPS)
                .create(ctx);
    }

    // orm

    std::filesystem::path ao_path, roughness_path, metallic_path;

    aiString ao_rel_path;
    if (assimp_material->GetTexture(aiTextureType_AMBIENT_OCCLUSION, 0, &ao_rel_path) == aiReturn_SUCCESS) {
        ao_path = base_path;
        ao_path /= ao_rel_path.C_Str();
        ao_path.make_preferred();
    }

    aiString roughness_rel_path;
    if (assimp_material->GetTexture(aiTextureType_DIFFUSE_ROUGHNESS, 0, &roughness_rel_path) == aiReturn_SUCCESS) {
        roughness_path = base_path;
        roughness_path /= roughness_rel_path.C_Str();
        roughness_path.make_preferred();
    }

    aiString metallic_rel_path;
    if (assimp_material->GetTexture(aiTextureType_METALNESS, 0, &metallic_rel_path) == aiReturn_SUCCESS) {
        metallic_path = base_path;
        metallic_path /= metallic_rel_path.C_Str();
        metallic_path.make_preferred();
    }

    auto orm_builder = TextureBuilder()
            .use_format(vk::Format::eR8G8B8A8Unorm)
            .with_flags(vk::TextureFlagBitsZRX::MIPMAPS)
            .with_swizzle({
                ao_path.empty() ? SwizzleComponent::MAX : SwizzleComponent::R,
                roughness_path.empty() ? SwizzleComponent::MAX : SwizzleComponent::G,
                metallic_path.empty() ? SwizzleComponent::ZERO : SwizzleComponent::B,
                SwizzleComponent::MAX,
            });

    if (ao_path.empty() && roughness_path.empty() && metallic_path.empty()) {
        orm_builder.from_swizzle_fill({1, 1, 1});
    } else if (!ao_path.empty() && (ao_path == roughness_path || ao_path == metallic_path)) {
        orm_builder.from_paths({ao_path});
    } else if (!roughness_path.empty() && (roughness_path == ao_path || roughness_path == metallic_path)) {
        orm_builder.from_paths({roughness_path});
    } else if (!metallic_path.empty() && (metallic_path == ao_path || metallic_path == roughness_path)) {
        orm_builder.from_paths({metallic_path});
    } else {
        orm_builder.as_separate_channels().from_paths({ao_path, roughness_path, metallic_path});
    }

    orm = orm_builder.create(ctx);
}

Model::Model(const RendererContext &ctx, const std::filesystem::path &path, const bool load_materials) {
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(
        path.string(),
        aiProcess_RemoveRedundantMaterials
        | aiProcess_FindInstances
        | aiProcess_OptimizeMeshes
        | aiProcess_OptimizeGraph
        | aiProcess_FixInfacingNormals
        | aiProcess_Triangulate
        | aiProcess_JoinIdenticalVertices
        | aiProcess_CalcTangentSpace
        | aiProcess_SortByPType
        | aiProcess_ImproveCacheLocality
        | aiProcess_ValidateDataStructure
    );

    if (!scene) {
        Logger::error(importer.GetErrorString());
    }

    if (load_materials) {
        constexpr size_t MAX_MATERIAL_COUNT = 32;
        if (scene->mNumMaterials > MAX_MATERIAL_COUNT) {
            Logger::error("Models with more than 32 materials are not supported");
        }

        for (size_t i = 0; i < scene->mNumMaterials; i++) {
            std::filesystem::path base_path = path.parent_path();
            materials.emplace_back(ctx, scene->mMaterials[i], base_path);
        }
    }

    for (size_t i = 0; i < scene->mNumMeshes; i++) {
        meshes.emplace_back(scene->mMeshes[i]);

        if (!load_materials) {
            meshes.back().material_id = 0;
        }
    }

    add_instances(scene->mRootNode, glm::identity<glm::mat4>());

    normalize_scale();

    create_buffers(ctx);
    // create_blas(ctx);
}

void Model::add_instances(const aiNode *node, const glm::mat4 &base_transform) {
    const glm::mat4 transform = base_transform * assimp_matrix_to_glm(node->mTransformation);

    for (size_t i = 0; i < node->mNumMeshes; i++) {
        meshes[node->mMeshes[i]].instances.push_back(transform);
    }

    for (size_t i = 0; i < node->mNumChildren; i++) {
        add_instances(node->mChildren[i], transform);
    }
}

vector<ModelVertex> Model::get_vertices() const {
    vector<ModelVertex> vertices;

    size_t total_size = 0;
    for (const auto &mesh: meshes) {
        total_size += mesh.vertices.size();
    }

    vertices.reserve(total_size);

    for (const auto &mesh: meshes) {
        vertices.insert(vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
    }

    return vertices;
}

vector<uint32_t> Model::get_indices() const {
    vector<uint32_t> indices;

    size_t total_size = 0;
    for (const auto &mesh: meshes) {
        total_size += mesh.indices.size();
    }

    indices.reserve(total_size);

    for (const auto &mesh: meshes) {
        indices.insert(indices.end(), mesh.indices.begin(), mesh.indices.end());
    }

    return indices;
}

vector<glm::mat4> Model::get_instance_transforms() const {
    vector<glm::mat4> result;

    size_t total_size = 0;
    for (const auto &mesh: meshes) {
        total_size += mesh.instances.size();
    }

    result.reserve(total_size);

    for (const auto &mesh: meshes) {
        result.insert(result.end(), mesh.instances.begin(), mesh.instances.end());
    }

    return result;
}

vector<MeshDescription> Model::get_mesh_descriptions() const {
    vector<MeshDescription> result;

    uint32_t index_offset  = 0;
    uint32_t vertex_offset = 0;

    for (const auto &mesh: meshes) {
        result.emplace_back(MeshDescription{
            .material_id = mesh.material_id,
            .vertex_offset = vertex_offset,
            .index_offset = index_offset,
        });

        index_offset += static_cast<uint32_t>(mesh.indices.size());
        vertex_offset += static_cast<std::int32_t>(mesh.vertices.size());
    }

    return result;
}

void Model::bind_buffers(const vk::raii::CommandBuffer &command_buffer) const {
    command_buffer.bindVertexBuffers(0, **vertex_buffer, {0});
    command_buffer.bindVertexBuffers(1, **instance_data_buffer, {0});
    command_buffer.bindIndexBuffer(**index_buffer, 0, vk::IndexType::eUint32);
}

void Model::create_buffers(const RendererContext &ctx) {
    constexpr auto ray_tracing_flags = vk::BufferUsageFlagBits::eStorageBuffer
                                       | vk::BufferUsageFlagBits::eShaderDeviceAddress
                                       | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;

    vertex_buffer = utils::buf::create_local_buffer(
        ctx,
        get_vertices(),
        vk::BufferUsageFlagBits::eVertexBuffer | ray_tracing_flags
    );

    instance_data_buffer = utils::buf::create_local_buffer(
        ctx,
        get_instance_transforms(),
        vk::BufferUsageFlagBits::eVertexBuffer | ray_tracing_flags
    );

    index_buffer = utils::buf::create_local_buffer(
        ctx,
        get_indices(),
        vk::BufferUsageFlagBits::eIndexBuffer | ray_tracing_flags
    );

    mesh_descriptions_buffer = utils::buf::create_local_buffer(
        ctx,
        get_mesh_descriptions(),
        ray_tracing_flags
    );
}

void Model::create_blas(const RendererContext &ctx) {
    const vk::DeviceAddress vertex_address = ctx.device->getBufferAddress({.buffer = **vertex_buffer});
    const vk::DeviceAddress index_address  = ctx.device->getBufferAddress({.buffer = **index_buffer});

    const uint32_t max_primitive_count = get_indices().size() / 3;

    const vk::AccelerationStructureGeometryTrianglesDataKHR geometry_triangles{
        .vertexFormat = vk::Format::eR32G32B32Sfloat,
        .vertexData = vertex_address,
        .vertexStride = sizeof(ModelVertex),
        .maxVertex = static_cast<uint32_t>(get_vertices().size() - 1),
        .indexType = vk::IndexType::eUint32,
        .indexData = index_address,
    };

    const vk::AccelerationStructureGeometryKHR geometry{
        .geometryType = vk::GeometryTypeKHR::eTriangles,
        .geometry = geometry_triangles,
        .flags = vk::GeometryFlagBitsKHR::eOpaque,
    };

    vk::AccelerationStructureBuildGeometryInfoKHR geometry_info{
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
        .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
        .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
        .geometryCount = 1u,
        .pGeometries = &geometry,
    };

    const vk::AccelerationStructureBuildRangeInfoKHR range_info{
        .primitiveCount = max_primitive_count,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };

    const auto build_sizes = ctx.device->getAccelerationStructureBuildSizesKHR(
        vk::AccelerationStructureBuildTypeKHR::eDevice,
        geometry_info,
        max_primitive_count
    );

    // scratch buffer creation

    const Buffer scratch_buffer{
        **ctx.allocator,
        build_sizes.buildScratchSize,
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    };

    geometry_info.scratchData = ctx.device->getBufferAddress({.buffer = *scratch_buffer});

    // acceleration structure creation

    const uint32_t acceleration_structure_size = build_sizes.accelerationStructureSize;

    auto blas_buffer = make_unique<Buffer>(
        **ctx.allocator,
        acceleration_structure_size,
        vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
        vk::MemoryPropertyFlagBits::eDeviceLocal
    );

    const vk::AccelerationStructureCreateInfoKHR as_create_info{
        .buffer = **blas_buffer,
        .size = acceleration_structure_size,
        .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
    };

    auto blas_handle = make_unique<vk::raii::AccelerationStructureKHR>(
        ctx.device->createAccelerationStructureKHR(as_create_info)
    );

    geometry_info.dstAccelerationStructure = **blas_handle;

    blas = make_unique<AccelerationStructure>(
        std::move(blas_handle),
        std::move(blas_buffer)
    );

    // todo - compact

    utils::cmd::do_single_time_commands(ctx, [&](const vk::raii::CommandBuffer &command_buffer) {
        command_buffer.buildAccelerationStructuresKHR(geometry_info, &range_info);
    });
}

void Model::normalize_scale() {
    constexpr float standard_scale = 10.0f;
    const float largest_distance = get_max_vertex_distance();
    const glm::mat4 scale_matrix = glm::scale(glm::identity<glm::mat4>(), glm::vec3(standard_scale / largest_distance));

    for (auto &mesh: meshes) {
        for (auto &transform: mesh.instances) {
            transform = scale_matrix * transform;
        }
    }
}

float Model::get_max_vertex_distance() const {
    float largest_distance = 0.0;

    for (const auto &mesh: meshes) {
        for (const auto &vertex: mesh.vertices) {
            for (const auto &transform: mesh.instances) {
                largest_distance = std::max(
                    largest_distance,
                    glm::length(glm::vec3(transform * glm::vec4(vertex.pos, 1.0)))
                );
            }
        }
    }

    return largest_distance;
}
}
