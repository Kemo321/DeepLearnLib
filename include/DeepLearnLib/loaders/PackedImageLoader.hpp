/**
 * @file PackedImageLoader.hpp
 * @brief MNIST loader for the packed @c DLIMG001 file.
 *
 * The file stores a little-endian header and then raw pixels, so a batch does
 * not open one image file per sample. Upload and prefetch match
 * @c ClassificationLoader: the next host batch is prepared while the current
 * one is consumed, and the last batch may be short.
 */

#pragma once

#include "DeepLearnLib/core/Batch.hpp"

#include <cstddef>
#include <cstdint>
#include <future>
#include <random>
#include <string>
#include <vector>

/**
 * @brief Batches from a single @c DLIMG001 blob.
 *
 * Pixels are scaled to float32 and laid out as NCHW. Labels are one-hot.
 * A file whose magic is not @c DLIMG001 is rejected.
 */
class PackedImageLoader
{
public:
    /**
     * @param bin_path Path to the packed file.
     * @param batch_size Maximum samples per batch.
     * @param shuffle Shuffle with the experiment seed at @c reset.
     */
    PackedImageLoader(std::string bin_path, int batch_size, bool shuffle = true);
    ~PackedImageLoader();

    PackedImageLoader(const PackedImageLoader&) = delete;
    auto operator=(const PackedImageLoader&) -> PackedImageLoader& = delete;
    PackedImageLoader(PackedImageLoader&&) = delete;
    auto operator=(PackedImageLoader&&) -> PackedImageLoader& = delete;

    /** Rewind and, when shuffling, draw a new sample order. */
    auto reset() -> void;

    /** True while another batch remains in this epoch. */
    [[nodiscard]] auto has_next() const -> bool;

    /**
     * @brief Take the next batch and upload it.
     * @param stream Stream used for the host-to-device copy.
     */
    auto get_batch(cudaStream_t stream = 0) -> Batch;

    /** Samples stored in the file. */
    [[nodiscard]] auto size() const -> std::size_t;

    /** Maximum samples requested per batch. */
    [[nodiscard]] auto batch_size() const -> int;

    /** One-hot width from the file header. */
    [[nodiscard]] auto num_classes() const -> int;

    /** Channels from the file header. */
    [[nodiscard]] auto channels() const -> int;

    /** Image height from the file header. */
    [[nodiscard]] auto height() const -> int;

    /** Image width from the file header. */
    [[nodiscard]] auto width() const -> int;

    /** Class names in one-hot order. The file stores none, so these are the indices @c "0", @c "1", ... */
    [[nodiscard]] auto class_names() const -> const std::vector<std::string>&;

    /** Integer label of sample @p index, before one-hot encoding. */
    [[nodiscard]] auto label_at(std::size_t index) const -> int;

    /** Copy one sample into @p nchw as NCHW float32 with a leading batch of 1. */
    auto copy_sample_float(std::size_t index, std::vector<float>& nchw) const -> void;

private:
    struct HostBatch
    {
        int n { 0 };
        std::vector<float> images;
        std::vector<float> targets;
    };

    auto take_indices() -> std::vector<std::size_t>;
    [[nodiscard]] auto decode_indices(const std::vector<std::size_t>& indices) const -> HostBatch;
    auto upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch;
    auto launch_prefetch() -> void;
    auto join_prefetch() -> void;

    int batch_size_;
    bool shuffle_;
    std::uint32_t sample_count_ { 0 };
    std::uint32_t channels_ { 0 };
    std::uint32_t height_ { 0 };
    std::uint32_t width_ { 0 };
    std::uint32_t num_classes_ { 0 };
    std::vector<std::uint8_t> pixels_;
    std::vector<std::uint8_t> labels_;
    std::vector<std::string> class_names_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ { 0 };
    std::mt19937 rng_;
    std::future<HostBatch> prefetch_;
};
