#pragma once

#define LOG_ERROR_WITH_FUNC(msg) Logger::error("{} [in function: {}]", (msg), __func__)
