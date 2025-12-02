module;

export module Cinder.Utils:UniqueHandle;

import std;

import Cinder.Globals;

export namespace zrx {
template <typename Tag>
class UniqueHandle {
public:
    using IDType = uint32_t;

private:
    static IDType next_free_handle_id;
    static IDType next_free_special_handle_id;
    IDType id;

    constexpr UniqueHandle(const IDType id_) : id(id_) {}

public:
    static constexpr UniqueHandle get_new() { return { next_free_handle_id++ }; }

    static constexpr UniqueHandle get_new_special() { return { next_free_special_handle_id-- }; }

    // almost never use this!!!
    // this is only here temporarily because we want to preserve assimp's IDs and keep the type safety
    // (yeah i know this is a code smell i will fix it in the future)
    static constexpr UniqueHandle get_unsafe(const IDType id_) { return { id_ }; }

    explicit operator uint32_t() const { return id; }

    constexpr auto operator<=>(const UniqueHandle&) const = default;
};
} // ~zrx
