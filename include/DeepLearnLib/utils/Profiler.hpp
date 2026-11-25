/**
 * @file Profiler.hpp
 * @brief Short GPU intervals and a process-wide VRAM reading.
 *
 * Epoch timing does not use this class. An epoch is a wall-clock interval
 * measured by the training program. @c Profiler is for a single GPU span
 * bracketed by CUDA events.
 */

#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

#include <cstddef>

/**
 * @brief Elapsed device time between @c start and @c stop.
 *
 * @c stop synchronizes the end event and returns milliseconds. One profiler
 * measures one interval at a time.
 */
class Profiler
{
public:
    Profiler();
    ~Profiler();

    Profiler(const Profiler&) = delete;
    auto operator=(const Profiler&) -> Profiler& = delete;
    Profiler(Profiler&&) = delete;
    auto operator=(Profiler&&) -> Profiler& = delete;

    /** Record the start event on the legacy default stream. */
    auto start() -> void;

    /**
     * @brief Record the stop event on the legacy default stream and wait for it.
     * @return Elapsed time in milliseconds.
     * @throws std::runtime_error if @c start was not called first.
     */
    [[nodiscard]] auto stop() -> float;

    /**
     * @brief Memory occupied on the current device, in mebibytes.
     *
     * This is @c cudaMemGetInfo: total minus free. The figure includes the
     * driver context, cuBLAS, and cuDNN, not only tensors allocated by the library.
     */
    [[nodiscard]] static auto get_vram_usage_mb() -> std::size_t;

private:
#if DEEPLEARNLIB_ENABLE_CUDA
    cudaEvent_t start_event_ { nullptr };
    cudaEvent_t stop_event_ { nullptr };
#endif
    bool running_ { false };
};
