#pragma once

#include <filesystem>
#include <vector>

#include "vertex.hpp"
#include "src/render/libs.hpp"
#include "src/render/globals.hpp"
#include "src/render/vk/accel-struct.hpp"

struct aiMaterial;
struct aiScene;
struct aiMesh;
struct aiNode;

namespace zrx {
struct RendererContext;
class Texture;
class Buffer;

struct Mesh {
    vector<ModelVertex> vertices;
    vector<uint32_t> indices;
    vector<glm::mat4> instances;
    uint32_t material_id;

    explicit Mesh(const aiMesh *assimp_mesh);
};

struct MeshDescription {
    uint32_t material_id;
    uint32_t vertex_offset;
    uint32_t index_offset;
};

struct Material {
    unique_ptr<Texture> base_color;
    unique_ptr<Texture> normal;
    unique_ptr<Texture> orm;

    Material() = default;

    explicit Material(const RendererContext &ctx, const aiMaterial *assimp_material,
                      const std::filesystem::path &base_path);
};

class Model {
    vector<Mesh> meshes;
    vector<Material> materials;

    unique_ptr<Buffer> vertex_buffer;
    unique_ptr<Buffer> instance_data_buffer;
    unique_ptr<Buffer> index_buffer;
    unique_ptr<Buffer> mesh_descriptions_buffer;

    unique_ptr<AccelerationStructure> blas;

public:
    explicit Model(const RendererContext &ctx, const std::filesystem::path &path, bool load_materials);

    void add_instances(const aiNode *node, const glm::mat4 &base_transform);

    [[nodiscard]] const vector<Mesh> &get_meshes() const { return meshes; }

    [[nodiscard]] const vector<Material> &get_materials() const { return materials; }

    [[nodiscard]] const Buffer &get_vertex_buffer() const { return *vertex_buffer; }

    [[nodiscard]] const Buffer &get_index_buffer() const { return *index_buffer; }

    [[nodiscard]] const Buffer &get_mesh_descriptions_buffer() const { return *mesh_descriptions_buffer; }

    [[nodiscard]] vector<ModelVertex> get_vertices() const;

    [[nodiscard]] vector<uint32_t> get_indices() const;

    [[nodiscard]] vector<glm::mat4> get_instance_transforms() const;

    [[nodiscard]] vector<MeshDescription> get_mesh_descriptions() const;

    [[nodiscard]] const vk::raii::AccelerationStructureKHR &get_blas() const { return **blas; }

    void bind_buffers(const vk::raii::CommandBuffer &command_buffer) const;

private:
    void normalize_scale();

    void create_buffers(const RendererContext &ctx);

    void create_blas(const RendererContext &ctx);

    [[nodiscard]] float get_max_vertex_distance() const;
};
} // zrx
