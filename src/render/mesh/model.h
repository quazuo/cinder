#pragma once

#include <filesystem>
#include <vector>

#include "vertex.h"
#include "src/render/libs.h"
#include "src/render/globals.h"
#include "src/render/vk/accel-struct.h"

struct RendererContext;
struct aiMaterial;
struct aiScene;
struct aiMesh;
struct aiNode;
class DescriptorSet;
class Texture;
class Buffer;

struct Mesh {
    std::vector<ModelVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<glm::mat4> instances;
    uint32_t materialID;

    explicit Mesh(const aiMesh *assimpMesh);
};

struct Material {
    unique_ptr<Texture> baseColor;
    unique_ptr<Texture> normal;
    unique_ptr<Texture> orm;

    Material() = default;

    explicit Material(const RendererContext &ctx, const aiMaterial *assimpMaterial,
                      const std::filesystem::path &basePath);
};

class Model {
    std::vector<Mesh> meshes;
    std::vector<Material> materials;

    unique_ptr<Buffer> vertexBuffer;
    unique_ptr<Buffer> instanceDataBuffer;
    unique_ptr<Buffer> indexBuffer;

    unique_ptr<AccelerationStructure> blas;

public:
    explicit Model(const RendererContext &ctx, const std::filesystem::path &path, bool loadMaterials);

    void addInstances(const aiNode *node, const glm::mat4 &baseTransform);

    [[nodiscard]] const std::vector<Mesh> &getMeshes() const { return meshes; }

    [[nodiscard]] const std::vector<Material> &getMaterials() const { return materials; }

    [[nodiscard]] std::vector<ModelVertex> getVertices() const;

    [[nodiscard]] std::vector<uint32_t> getIndices() const;

    [[nodiscard]] std::vector<glm::mat4> getInstanceTransforms() const;

    [[nodiscard]] const vk::raii::AccelerationStructureKHR& getBLAS() const { return **blas; }

    void bindBuffers(const vk::raii::CommandBuffer& commandBuffer) const;

private:
    void normalizeScale();

    void createBuffers(const RendererContext &ctx);

    void createBLAS(const RendererContext &ctx);

    [[nodiscard]] float getMaxVertexDistance() const;
};
