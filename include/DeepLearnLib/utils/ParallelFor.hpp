/**
 * @file ParallelFor.hpp
 * @brief Small host-side parallel loop used while decoding a batch.
 */

#pragma once

#include <algorithm>
#include <exception>
#include <future>
#include <thread>
#include <vector>

namespace dl
{

/**
 * @brief How many workers to spawn for @p work_items independent jobs.
 *
 * The count is at most 16, at most the hardware concurrency, and at most
 * @p work_items. One item or fewer uses one worker. Unknown hardware
 * concurrency is treated as 8.
 */
inline auto parallel_worker_count(int work_items) -> int
{
    if (work_items <= 1)
    {
        return 1;
    }
    const unsigned hardware = std::thread::hardware_concurrency();
    int cap = (hardware == 0U) ? 8 : static_cast<int>(hardware);
    cap = std::min(cap, 16);
    return std::min(work_items, cap);
}

/**
 * @brief Invoke @p fn once for every index in @c [0, count).
 *
 * Work is striped across workers. The first exception thrown by a worker is
 * rethrown after the remaining workers have finished. @p fn must be safe to
 * call concurrently on distinct indices.
 */
template <typename Fn>
auto parallel_for(int count, Fn&& fn) -> void
{
    if (count <= 0)
    {
        return;
    }
    if (count == 1)
    {
        fn(0);
        return;
    }

    const int workers = parallel_worker_count(count);
    std::vector<std::future<void>> tasks;
    tasks.reserve(static_cast<std::size_t>(workers));
    for (int worker = 0; worker < workers; ++worker)
    {
        tasks.emplace_back(std::async(std::launch::async,
            [worker, workers, count, &fn]()
            {
                for (int index = worker; index < count; index += workers)
                {
                    fn(index);
                }
            }));
    }

    std::exception_ptr first_error;
    for (auto& task : tasks)
    {
        try
        {
            task.get();
        }
        catch (...)
        {
            if (!first_error)
            {
                first_error = std::current_exception();
            }
        }
    }
    if (first_error)
    {
        std::rethrow_exception(first_error);
    }
}

}
