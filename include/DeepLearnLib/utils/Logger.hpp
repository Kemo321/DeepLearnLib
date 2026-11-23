/**
 * @file Logger.hpp
 * @brief Process-wide spdlog logger used by the CUDA status checks.
 */

#pragma once

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

namespace dl
{

/**
 * @brief Write @p message to the process logger.
 *
 * Called from the CUDA, cuBLAS, and cuDNN status checks before they throw.
 * This does not replace the metrics CSV written by a training program.
 */
auto log_error_message(const std::string& message) -> void;

/**
 * @brief Singleton spdlog logger for the library.
 *
 * Copy and move are deleted. Call @c get when a site needs the spdlog handle
 * directly. Most call sites should use @c log_error_message.
 */
class Logger
{
public:
    /** The singleton. Constructed on first use. */
    static auto instance() -> Logger&;

    /** Shared spdlog handle owned by the singleton. */
    static auto get() -> std::shared_ptr<spdlog::logger>;

    Logger(const Logger&) = delete;
    auto operator=(const Logger&) -> Logger& = delete;
    Logger(Logger&&) = delete;
    auto operator=(Logger&&) -> Logger& = delete;

private:
    Logger();
    ~Logger();

    std::shared_ptr<spdlog::logger> logger_;
};

}
