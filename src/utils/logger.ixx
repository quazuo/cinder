module;

export module Cinder.Utils:Logger;

import imgui;
import std;

import Cinder.Globals;

export namespace zrx {
enum class LogLevel : uint8_t {
    FATAL_ERROR,
    WARNING,
    INFO,
    DEBUG,
};

class Logger {
public:
    static inline bool ENABLE_ERROR_LOGS    = true;
    static inline bool ENABLE_WARNING_LOGS  = true;
    static inline bool ENABLE_INFO_LOGS     = true;
    static inline bool ENABLE_DEBUG_LOGS    = true;

    static void render_gui_section() {
        constexpr auto section_flags = ImGuiTreeNodeFlags_DefaultOpen;

        if (ImGui::CollapsingHeader("Logging ", section_flags)) {
            ImGui::Checkbox("Error",    &ENABLE_ERROR_LOGS);
            ImGui::Checkbox("Warning",  &ENABLE_WARNING_LOGS);
            ImGui::Checkbox("Info",     &ENABLE_INFO_LOGS);
            ImGui::Checkbox("Debug",    &ENABLE_DEBUG_LOGS);
        }
    }

    template<typename T, typename... Ts>
    static void error(const T& fmt_str, Ts... args) {
        if (!ENABLE_ERROR_LOGS) return;
        log(std::cout, LogLevel::FATAL_ERROR, fmt_str, args...);
        std::stringstream ss;
        ss << "FATAL_ERROR: " << std::vformat(fmt_str, std::make_format_args(args...));
        throw std::runtime_error(ss.str());
    }

    template<typename T, typename... Ts>
    static void warning(const T& fmt_str, Ts... args) {
        if (!ENABLE_WARNING_LOGS) return;
        log(std::cout, LogLevel::WARNING, fmt_str, args...);
    }

    template<typename T, typename... Ts>
    static void info(const T& fmt_str, Ts... args) {
        if (!ENABLE_INFO_LOGS) return;
        log(std::cout, LogLevel::INFO, fmt_str, args...);
    }

    template<typename T, typename... Ts>
    static void debug(const T& fmt_str, Ts... args) {
#ifndef NDEBUG
        if (!ENABLE_DEBUG_LOGS) return;
        log(std::cout, LogLevel::DEBUG, fmt_str, args...);
#endif
    }

private:
    template<typename T, typename... Ts>
    static void log(std::ostream& stream, const LogLevel level, const T& fmt_str, Ts... args) {
        stream << "[LOG / " << to_string(level) << "] ";
        stream << std::vformat(fmt_str, std::make_format_args(args...));
        stream << "\n";
    }

    static auto to_string(const LogLevel level) -> string {
        switch (level) {
            case LogLevel::FATAL_ERROR:
                return "FATAL ERROR";
            case LogLevel::WARNING:
                return "WARNING";
            case LogLevel::INFO:
                return "INFO";
            case LogLevel::DEBUG:
                return "DEBUG";
            default:
                throw std::runtime_error("missing path in Logger::to_string");
        }
    }
};
} // zrx
