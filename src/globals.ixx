module;

#include <cinttypes>

export module Cinder.Globals;

import std;

export {
    using std::unique_ptr;
    using std::make_unique;
    using std::shared_ptr;
    using std::make_shared;
    using std::reference_wrapper;
    using std::vector;
    using std::array;
    using std::map;
    using std::set;
    using std::string;
    using std::optional;
    using std::pair;
    using std::numeric_limits;
    using std::variant;

    using std::uint64_t;
    using std::uint32_t;
    using std::uint16_t;
    using std::uint8_t;

    using std::int64_t;
    using std::int32_t;
    using std::int16_t;
    using std::int8_t;

    using ResourceHandle = uint32_t;
    using ResourceHandleArray = vector<ResourceHandle>;
    using BindlessHandle = uint32_t;
}
