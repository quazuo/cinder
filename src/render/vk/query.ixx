module;

export module Cinder.Render.Vulkan:Query;

import vulkan_hpp;
import std;

import Cinder.Utils;
import Cinder.Globals;
import :Buffer;
import :Image;
import :Context;

export namespace zrx {
class QueryPool {
    unique_ptr<vk::raii::QueryPool> query_pool;
    uint32_t query_count;

public:
    QueryPool(const RendererContext& ctx, vk::QueryType query_type, uint32_t query_count,
              vk::QueryPipelineStatisticFlags pipeline_statistics = {});

    auto operator*() const -> const vk::raii::QueryPool& { return *query_pool; }

    auto get_results() const -> vector<uint64_t>;
};
} // zrx
