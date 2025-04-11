#pragma once

#include <iostream>
#include <sstream>
#include <format>

#include "src/render/globals.hpp"

namespace zrx {
enum class LogLevel : uint8_t {
    FATAL_ERROR,
    WARNING,
    INFO,
    DEBUG,
};

class Logger {
public:
    template<typename T, typename... Ts>
    static void error(const T& fmt_str, Ts... args) {
        std::stringstream ss;
        ss << "FATAL_ERROR: " << std::vformat(fmt_str, std::make_format_args(args...));
       throw std::runtime_error(ss.str());
    }

    template<typename T, typename... Ts>
    static void warning(const T& fmt_str, Ts... args) {
        log(std::cerr, LogLevel::WARNING, fmt_str, args...);
    }

    template<typename T, typename... Ts>
    static void info(const T& fmt_str, Ts... args) {
        log(std::cout, LogLevel::INFO, fmt_str, args...);
    }

    template<typename T, typename... Ts>
    static void debug(const T& fmt_str, Ts... args) {
#ifndef NDEBUG
        log(std::cout, LogLevel::DEBUG, fmt_str, args...);
#endif
    }

private:
    template<typename T, typename... Ts>
    static void log(std::ostream& stream, const LogLevel level, const T& fmt_str, Ts... args) {
        stream << "[LOG / " << to_string(level) << "]\n";
        stream << std::vformat(fmt_str, std::make_format_args(args...));
        stream << "\n\n";
    }

    [[nodiscard]] static string to_string(const LogLevel level) {
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
