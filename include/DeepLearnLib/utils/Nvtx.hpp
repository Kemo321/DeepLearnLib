/**
 * @file Nvtx.hpp
 * @brief RAII range for NVIDIA Tools Extension, or a no-op when the header is absent.
 *
 * Forward, backward, and the optimizer step push a range so a Nsight trace can
 * name those intervals. A build without NVTX compiles the same calls as empty.
 */

#pragma once

#ifndef DEEPLEARNLIB_ENABLE_CUDA
#define DEEPLEARNLIB_ENABLE_CUDA 1
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif

#if DEEPLEARNLIB_ENABLE_CUDA
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define DEEPLEARNLIB_HAS_NVTX 1
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#define DEEPLEARNLIB_HAS_NVTX 1
#endif
#endif
#endif

#ifndef DEEPLEARNLIB_HAS_NVTX
#define DEEPLEARNLIB_HAS_NVTX 0
#endif

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace dl
{

/**
 * @brief Push an NVTX range for the lifetime of this object.
 *
 * Construct at the start of a scope. The destructor pops the range.
 * With NVTX disabled the constructor ignores @p name.
 */
class NvtxRange
{
public:
    explicit NvtxRange(const char* name)
    {
#if DEEPLEARNLIB_HAS_NVTX
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    ~NvtxRange()
    {
#if DEEPLEARNLIB_HAS_NVTX
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    auto operator=(const NvtxRange&) -> NvtxRange& = delete;
    NvtxRange(NvtxRange&&) = delete;
    auto operator=(NvtxRange&&) -> NvtxRange& = delete;
};

}
