module;

export module Cinder.Utils:Concepts;

import std;

export namespace zrx {

template <typename T, typename... Ts>
concept is_one_of = (std::same_as<std::remove_cvref_t<T>, Ts> || ...);

} // zrx