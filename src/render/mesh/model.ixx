module;

export module Cinder.Render.Mesh:Model;

import assimp;
import std;

import Cinder.Render.Vulkan;
import Cinder.Globals;
import :Vertex;

export namespace zrx {
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

    auto get_meshes() const -> const vector<Mesh>& { return meshes; }

    auto get_materials() const -> const vector<Material>& { return materials; }

    auto get_vertex_buffer() const -> const Buffer& { return *vertex_buffer; }

    auto get_index_buffer() const -> const Buffer& { return *index_buffer; }

    auto get_mesh_descriptions_buffer() const -> const Buffer& { return *mesh_descriptions_buffer; }

    auto get_blas() const -> const AccelerationStructure& { return *blas; }

    auto get_vertices() const -> vector<ModelVertex>;

    auto get_indices() const -> vector<uint32_t>;

    auto get_instance_transforms() const -> vector<glm::mat4>;

    auto get_mesh_descriptions() const -> vector<MeshDescription>;

    void bind_buffers(const vk::raii::CommandBuffer &command_buffer) const;

private:
    void normalize_scale();

    void create_buffers(const RendererContext &ctx);

    void create_blas(const RendererContext &ctx);

    auto get_max_vertex_distance() const -> float;
};
} // zrx
