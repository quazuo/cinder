#pragma once

#include <memory>

using std::unique_ptr;
using std::make_unique;
using std::shared_ptr;
using std::make_shared;
using std::reference_wrapper;
using std::vector;

using std::uint32_t;
using std::uint16_t;
using std::uint8_t;

using std::int32_t;
using std::int16_t;
using std::int8_t;

using ResourceHandle = uint32_t;
using ResourceHandleArray = vector<ResourceHandle>;
