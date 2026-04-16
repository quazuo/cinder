module;

export module Cinder.Render.Graph;

export import :Node;
export import :Resource;
export import :ResourceManager;

import std;

import Cinder.Render.Vulkan;
import Cinder.Render.Mesh;
import Cinder.Globals;

export namespace zrx {
struct FrameBeginActionContext {
    reference_wrapper<ResourceManager> resource_manager;
};

using FrameBeginCallback = std::function<void(const FrameBeginActionContext &)>;

class RenderGraph {
    map<RenderNodeHandle, RenderNode> nodes_; // todo - replace by vector or smth else
    map<RenderNodeHandle, set<RenderNodeHandle> > dependency_graph;

    set<RenderNodeHandle> nodes_writing_to_final;
    set<ResourceHandle> produced_resources;

public:
    const auto &nodes() const { return nodes_; }

    auto get_topo_sorted() const -> vector<RenderNodeHandle>;

    auto get_partitioned() const -> vector<vector<RenderNodeHandle>>;

    auto add_node(const RenderNodeGraphics &node) -> RenderNodeHandle;
    auto add_node(const RenderNodeCompute &node)  -> RenderNodeHandle;

    /// Adds multiple nodes at once, connected sequentially via explicit dependencies.
    auto add_nodes_sequential(vector<RenderNode> nodes) -> vector<RenderNodeHandle>;

private:
    void add_new_dependencies(RenderNodeHandle new_handle);

    void cycles_helper(RenderNodeHandle handle, set<RenderNodeHandle> &discovered, set<RenderNodeHandle> &finished) const;

    void check_dependency_cycles() const;
};
} // zrx
