#pragma once

#include <iostream>
#include <sstream>

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
    template<typename... Ts>
    static void error(Ts... args) {
        std::stringstream ss;
        ss << "FATAL_ERROR: ";
        (ss << ... << args);
       throw std::runtime_error(ss.str());
    }

    template<typename... Ts>
    static void warning(Ts... args) {
        log(std::cerr, LogLevel::WARNING, args...);
    }

    template<typename... Ts>
    static void info(Ts... args) {
        log(std::cout, LogLevel::INFO, args...);
    }

    template<typename... Ts>
    static void debug(Ts... args) {
#ifndef NDEBUG
        log(std::cout, LogLevel::DEBUG, args...);
#endif
    }

private:
    template<typename... Ts>
    static void log(std::ostream& stream, const LogLevel level, Ts... args) {
        stream << "[LOG / " << to_string(level) << "]\n";
        (stream << ... << args);
        stream << "\n\n";
    }

    [[nodiscard]] static std::string to_string(const LogLevel level) {
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
