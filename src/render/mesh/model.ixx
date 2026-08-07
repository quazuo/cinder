module;

export module Cinder.Render.Mesh:Model;

import assimp;
import std;
import glm;

import Cinder.Render.Vulkan;
import Cinder.Render.Graph;
import Cinder.Render;
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

#include "src/render/glsl_to_cpp.inl"
#include "shaders/utils/model.glsl" // struct MeshDescription { ... }

struct MaterialDescPack {
    optional<ExternalTextureResourceDesc> base_color;
    optional<ExternalTextureResourceDesc> normal;
    optional<ExternalTextureResourceDesc> orm;

    MaterialDescPack() = default;

    explicit MaterialDescPack(const aiMaterial *assimp_material, const std::filesystem::path &base_path);
};

class Model {
    string name;

    vector<Mesh> meshes;
    vector<MaterialDescPack> material_desc_packs;

    vector<ModelVertex> vertices;
    vector<uint32_t> indices;
    vector<glm::mat4> instance_transforms;
    vector<MeshDescription> mesh_descriptions;

    optional<ResourceHandle> vertex_buffer;
    optional<ResourceHandle> instance_data_buffer;
    optional<ResourceHandle> index_buffer;
    optional<ResourceHandle> mesh_descriptions_buffer;

    // unique_ptr<AccelerationStructure> blas;

public:
    Model(string&& name, const std::filesystem::path &path, bool load_materials);

    void add_instances(const aiNode *node, const glm::mat4 &base_transform);

    auto get_meshes() const -> const vector<Mesh>& { return meshes; }

    auto get_mesh_descriptions() const -> const vector<MeshDescription>& { return mesh_descriptions; }

    auto get_vertex_buffer() const -> ResourceHandle { return *vertex_buffer; }

    auto get_instance_data_buffer() const -> ResourceHandle { return *instance_data_buffer; }

    auto get_index_buffer() const -> ResourceHandle { return *index_buffer; }

    auto get_mesh_descriptions_buffer() const -> ResourceHandle { return *mesh_descriptions_buffer; }

    // auto get_blas() const -> const AccelerationStructure& { return *blas; }

    void register_render_graph_resources(const VulkanRenderer& renderer);

private:
    auto get_vertices() const -> vector<ModelVertex>;

    auto get_indices() const -> vector<uint32_t>;

    auto get_instance_transforms() const -> vector<glm::mat4>;

    void normalize_scale();

    // void create_blas(const RendererContext &ctx);

    auto get_max_vertex_distance() const -> float;
};
} // zrx
