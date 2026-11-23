/**
 * @file Batch.hpp
 * @brief One supervised mini-batch.
 */

#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

/**
 * @brief Inputs and one-hot targets for a single training step.
 *
 * Image batches use NCHW. Tabular batches use @c [batch, features].
 * GRU batches use @c [batch, time, features]. Targets are one-hot float32
 * with the same batch dimension as @c images.
 */
struct Batch
{
    /** Samples for this step, already on the device used by the loader. */
    dl::Tensor images;

    /** One-hot labels, one row per sample. */
    dl::Tensor targets;
};
