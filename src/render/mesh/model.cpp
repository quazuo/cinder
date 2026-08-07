module;

module Cinder.Render.Mesh;

import Cinder.Utils;
import Cinder.Render;
import Cinder.Render.Vulkan;
import :Vertex;

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
    for (size_t vert_idx = 0; vert_idx < assimp_mesh->mNumVertices; vert_idx++) {
        ModelVertex vertex;

        if (assimp_mesh->HasPositions()) {
            vertex.pos = assimp_vec_to_glm(assimp_mesh->mVertices[vert_idx]);
        }

        if (assimp_mesh->HasTextureCoords(0)) {
            vertex.tex_coord = {
                assimp_mesh->mTextureCoords[0][vert_idx].x,
                1.0f - assimp_mesh->mTextureCoords[0][vert_idx].y
            };
        }

        if (assimp_mesh->HasTangentsAndBitangents()) {
            vertex.normal = assimp_vec_to_glm(assimp_mesh->mNormals[vert_idx]);
        }

        if (assimp_mesh->HasTangentsAndBitangents()) {
            vertex.tangent   = assimp_vec_to_glm(assimp_mesh->mTangents[vert_idx]);
            vertex.bitangent = assimp_vec_to_glm(assimp_mesh->mBitangents[vert_idx]);
        }

        vertices.push_back(vertex);
    }

    for (size_t face_idx = 0; face_idx < assimp_mesh->mNumFaces; face_idx++) {
        const auto &face = assimp_mesh->mFaces[face_idx];

        for (uint32_t i = 0; i < face.mNumIndices; i++) {
            indices.push_back(face.mIndices[i]);
        }
    }
}

MaterialDescPack::MaterialDescPack(const aiMaterial *assimp_material, const std::filesystem::path &base_path) {
    static uint32_t mat_idx = 0;

    // base color

    aiString base_color_rel_path;
    aiReturn result = assimp_material->GetTexture(aiTextureType_BASE_COLOR, 0, &base_color_rel_path);

    if (result == aiReturn_SUCCESS) {
        auto path = base_path;
        path /= base_color_rel_path.C_Str();
        path.make_preferred();

        base_color = ExternalTextureResourceDesc {
            .name = std::format("tex-base-color#{}@{}", mat_idx, assimp_material->GetName().C_Str()),
            .paths = { path },
            .format = vk::Format::eR8G8B8A8Srgb,
        };
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

        normal = ExternalTextureResourceDesc {
            .name = std::format("tex-normal#{}@{}", mat_idx, assimp_material->GetName().C_Str()),
            .paths = { path },
            .format = vk::Format::eR8G8B8A8Unorm,
        };
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

    ExternalTextureResourceDesc orm_desc {
        .name = std::format("tex-orm#{}@{}", mat_idx, assimp_material->GetName().C_Str()),
        .format = vk::Format::eR8G8B8A8Unorm,
        .swizzle = SwizzleDesc {
            ao_path.empty() ? SwizzleComponent::MAX : SwizzleComponent::R,
            roughness_path.empty() ? SwizzleComponent::MAX : SwizzleComponent::G,
            metallic_path.empty() ? SwizzleComponent::ZERO : SwizzleComponent::B,
            SwizzleComponent::MAX,
        }
    };

    if (ao_path.empty() && roughness_path.empty() && metallic_path.empty()) {
        orm_desc.paths = {};
    } else if (!ao_path.empty() && (ao_path == roughness_path || ao_path == metallic_path)) {
        orm_desc.paths = {ao_path};
    } else if (!roughness_path.empty() && (roughness_path == ao_path || roughness_path == metallic_path)) {
        orm_desc.paths = {roughness_path};
    } else if (!metallic_path.empty() && (metallic_path == ao_path || metallic_path == roughness_path)) {
        orm_desc.paths = {metallic_path};
    } else {
        orm_desc.paths = {ao_path, roughness_path, metallic_path};
    }

    if (!orm_desc.paths.empty()) {
        orm = orm_desc;
    }

    mat_idx++;
}

Model::Model(string&& name, const std::filesystem::path &path, const bool load_materials) : name(name) {
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(
        path.string(),
        aiProcess_RemoveRedundantMaterials
        | aiProcess_FindInstances
        | aiProcess_OptimizeMeshes
        | aiProcess_OptimizeGraph
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
        for (size_t i = 0; i < scene->mNumMaterials; i++) {
            std::filesystem::path base_path = path.parent_path();
            material_desc_packs.emplace_back(scene->mMaterials[i], base_path);
        }
    }

    uint32_t index_offset  = 0;
    uint32_t vertex_offset = 0;

    for (size_t i = 0; i < scene->mNumMeshes; i++) {
        meshes.emplace_back(scene->mMeshes[i]);

        mesh_descriptions.emplace_back(MeshDescription {
            .vertex_offset = vertex_offset,
            .index_offset  = index_offset,
            .base_color_id = PLACEHOLDER_BINDLESS_HANDLE,
            .normal_id     = PLACEHOLDER_BINDLESS_HANDLE,
            .orm_id        = PLACEHOLDER_BINDLESS_HANDLE,
        });

        index_offset += static_cast<uint32_t>(meshes[i].indices.size());
        vertex_offset += static_cast<std::int32_t>(meshes[i].vertices.size());
    }

    add_instances(scene->mRootNode, glm::gtc::identity<glm::mat4>());
    normalize_scale();
    // create_blas(ctx);

    vertices = get_vertices();
    indices = get_indices();
    instance_transforms = get_instance_transforms();
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

void Model::register_render_graph_resources(const VulkanRenderer& renderer) {
    vertex_buffer = renderer.register_resource(VertexBufferResourceDesc{
        .name = std::format("model-vb@{}", name),
        .size = vertices.size() * sizeof(decltype(vertices[0])),
        .data = vertices.data(),
    });

    instance_data_buffer = renderer.register_resource(VertexBufferResourceDesc{
        .name = std::format("model-idb@{}", name),
        .size = instance_transforms.size() * sizeof(decltype(instance_transforms[0])),
        .data = instance_transforms.data(),
    });

    index_buffer = renderer.register_resource(IndexBufferResourceDesc{
        .name = std::format("model-ib@{}", name),
        .size = indices.size() * sizeof(decltype(indices[0])),
        .data = indices.data(),
    });

    mesh_descriptions_buffer = renderer.register_resource(UniformBufferResourceDesc{
        .name = std::format("model-md@{}", name),
        .size = mesh_descriptions.size() * sizeof(decltype(mesh_descriptions[0])),
    });

    for (const auto &[mesh, mesh_desc] : std::views::zip(meshes, mesh_descriptions)) {
        const MaterialDescPack& mdp = material_desc_packs[mesh.material_id];

        if (mdp.base_color) {
            mesh_desc.base_color_id = renderer.get_bindless_handle(renderer.register_resource(*mdp.base_color));
        }

        if (mdp.normal) {
            mesh_desc.normal_id = renderer.get_bindless_handle(renderer.register_resource(*mdp.normal));
        }

        if (mdp.orm) {
            mesh_desc.orm_id = renderer.get_bindless_handle(renderer.register_resource(*mdp.orm));
        }
    }
}

// void Model::create_blas(const RendererContext &ctx) {
//     // todo - convert some of the stuff to VMA calls
//
//     const vk::DeviceAddress vertex_address = ctx.device->getBufferAddress({.buffer = **vertex_buffer});
//     const vk::DeviceAddress index_address  = ctx.device->getBufferAddress({.buffer = **index_buffer});
//
//     const uint32_t max_primitive_count = get_indices().size() / 3;
//
//     const vk::AccelerationStructureGeometryTrianglesDataKHR geometry_triangles{
//         .vertexFormat = vk::Format::eR32G32B32Sfloat,
//         .vertexData = vertex_address,
//         .vertexStride = sizeof(ModelVertex),
//         .maxVertex = static_cast<uint32_t>(get_vertices().size() - 1),
//         .indexType = vk::IndexType::eUint32,
//         .indexData = index_address,
//     };
//
//     const vk::AccelerationStructureGeometryKHR geometry{
//         .geometryType = vk::GeometryTypeKHR::eTriangles,
//         .geometry = geometry_triangles,
//         .flags = vk::GeometryFlagBitsKHR::eOpaque,
//     };
//
//     vk::AccelerationStructureBuildGeometryInfoKHR geometry_info{
//         .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
//         .flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace,
//         .mode = vk::BuildAccelerationStructureModeKHR::eBuild,
//         .geometryCount = 1u,
//         .pGeometries = &geometry,
//     };
//
//     const vk::AccelerationStructureBuildRangeInfoKHR range_info{
//         .primitiveCount = max_primitive_count,
//         .primitiveOffset = 0,
//         .firstVertex = 0,
//         .transformOffset = 0,
//     };
//
//     const auto build_sizes = ctx.device->getAccelerationStructureBuildSizesKHR(
//         vk::AccelerationStructureBuildTypeKHR::eDevice,
//         geometry_info,
//         max_primitive_count
//     );
//
//     // scratch buffer creation
//
//     const Buffer scratch_buffer{
//         ctx,
//         build_sizes.buildScratchSize,
//         vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer,
//         vk::MemoryPropertyFlagBits::eDeviceLocal
//     };
//
//     geometry_info.scratchData = ctx.device->getBufferAddress({.buffer = *scratch_buffer});
//
//     // acceleration structure creation
//
//     const uint32_t acceleration_structure_size = build_sizes.accelerationStructureSize;
//
//     auto blas_buffer = make_unique<Buffer>(
//         ctx,
//         acceleration_structure_size,
//         vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR,
//         vk::MemoryPropertyFlagBits::eDeviceLocal
//     );
//
//     const vk::AccelerationStructureCreateInfoKHR as_create_info{
//         .buffer = **blas_buffer,
//         .size = acceleration_structure_size,
//         .type = vk::AccelerationStructureTypeKHR::eBottomLevel,
//     };
//
//     auto blas_handle = make_unique<vk::raii::AccelerationStructureKHR>(
//         ctx.device->createAccelerationStructureKHR(as_create_info)
//     );
//
//     geometry_info.dstAccelerationStructure = **blas_handle;
//
//     blas = make_unique<AccelerationStructure>(
//         std::move(blas_handle),
//         std::move(blas_buffer)
//     );
//
//     // todo - compact
//
//     utils::cmd::do_single_time_commands(ctx, [&](const vk::raii::CommandBuffer &command_buffer) {
//         command_buffer.buildAccelerationStructuresKHR(geometry_info, &range_info);
//     });
// }

void Model::normalize_scale() {
    constexpr float standard_scale = 10.0f;
    const float largest_distance = get_max_vertex_distance();
    const glm::mat4 scale_matrix = glm::gtc::scale(glm::gtc::identity<glm::mat4>(), glm::vec3(standard_scale / largest_distance));

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
