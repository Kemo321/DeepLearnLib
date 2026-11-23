/**
 * @file SafeMath.hpp
 * @brief Host and device helpers that keep divisions away from zero.
 *
 * The cross-entropy kernels use the same epsilon when they clamp a probability
 * and when they divide by the batch size.
 */

#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define DL_HOST_DEVICE __host__ __device__
#else
#define DL_HOST_DEVICE
#endif

namespace dl
{

/** Added to a denominator, and the lower bound of a probability passed to a logarithm. */
constexpr float kSafeEps = 1e-7F;

/** Default gradient-clip bound. Zero means clipping is off. */
constexpr float kDefaultGradientClip = 0.0F;

/** @return @c numerator / (denominator + kSafeEps). */
DL_HOST_DEVICE inline auto safe_div(float numerator, float denominator) -> float
{
    return numerator / (denominator + kSafeEps);
}

/** @return @c 1 / max(denominator, kSafeEps). */
DL_HOST_DEVICE inline auto safe_inv(float denominator) -> float
{
    return 1.0F / fmaxf(denominator, kSafeEps);
}

/** Clamp @p probability to @c [kSafeEps, 1]. */
DL_HOST_DEVICE inline auto clamp_unit(float probability) -> float
{
    return fminf(fmaxf(probability, kSafeEps), 1.0F);
}

}
