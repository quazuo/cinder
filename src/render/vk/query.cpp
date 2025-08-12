module;

module Cinder.Render.Vulkan;



namespace zrx {
QueryPool::QueryPool(const RendererContext& ctx, const vk::QueryType query_type, const uint32_t query_count,
          const vk::QueryPipelineStatisticFlags pipeline_statistics) : query_count(query_count) {
    const vk::QueryPoolCreateInfo query_pool_create_info {
        .queryType = query_type,
        .queryCount = query_count,
        .pipelineStatistics = pipeline_statistics
    };

    query_pool = make_unique<vk::raii::QueryPool>(*ctx.device, query_pool_create_info);
    query_pool->reset(0, query_count);
}

auto QueryPool::get_results() const -> vector<uint64_t> {
    auto [ret_code, results] = query_pool->getResults<uint64_t>(
        0,
        query_count,
        query_count * sizeof(uint64_t),
        sizeof(uint64_t),
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait
    );

    if (ret_code != vk::Result::eSuccess) {
        Logger::warning("error [{}] in QueryPool::get_results", vk::to_string(ret_code));
    }

    query_pool->reset(0, query_count);

    return results;
}
} // zrx