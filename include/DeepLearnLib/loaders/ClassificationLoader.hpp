/**
 * @file ClassificationLoader.hpp
 * @brief Image-folder loader for CIFAR-10-style class directories.
 *
 * Samples are decoded with OpenCV on the host, arranged as NCHW, and copied
 * to the device. The next batch is decoded on a helper thread while the
 * current batch is in use. The last batch of an epoch may be shorter than
 * @c batch_size. It is not dropped and it is not padded.
 */

#pragma once

#include "DeepLearnLib/core/Batch.hpp"

#include <cstddef>
#include <future>
#include <random>
#include <string>
#include <utility>
#include <vector>

/**
 * @brief One epoch over a class directory of image files.
 *
 * The class subdirectories are discovered unless @c class_names is provided.
 * When names are provided, a missing folder still occupies a one-hot column.
 * Shuffle uses the Fisher-Yates helper and the experiment seed.
 */
class ClassificationLoader
{
public:
    /**
     * @param dataset_root Directory that contains the split folders.
     * @param split Split folder name, for example @c "train" or @c "test".
     * @param batch_size Maximum samples per batch.
     * @param image_size Height and width after decoding. The default matches CIFAR-10.
     * @param shuffle Shuffle the sample order at @c reset.
     * @param class_names Fixed label order. Empty means discover directories.
     */
    ClassificationLoader(std::string dataset_root, std::string split, int batch_size, int image_size = 32,
        bool shuffle = true, std::vector<std::string> class_names = {});
    ~ClassificationLoader();

    ClassificationLoader(const ClassificationLoader&) = delete;
    auto operator=(const ClassificationLoader&) -> ClassificationLoader& = delete;
    ClassificationLoader(ClassificationLoader&&) = delete;
    auto operator=(ClassificationLoader&&) -> ClassificationLoader& = delete;

    /** Rewind to the first batch and, when shuffling, draw a new order. */
    auto reset() -> void;

    /** True while @c get_batch can still return a non-empty batch. */
    [[nodiscard]] auto has_next() const -> bool;

    /**
     * @brief Decode, or take the prefetched host batch, and upload it.
     * @param stream Stream used for the host-to-device copy.
     */
    auto get_batch(cudaStream_t stream = 0) -> Batch;

    /** Number of samples in this split. */
    [[nodiscard]] auto size() const -> std::size_t;

    /** Maximum samples requested per batch. The last batch may be smaller. */
    [[nodiscard]] auto batch_size() const -> int;

    /** One-hot width. */
    [[nodiscard]] auto num_classes() const -> int;

    /** Spatial size passed to the constructor. */
    [[nodiscard]] auto image_size() const -> int;

    /** Class names in one-hot column order. */
    [[nodiscard]] auto class_names() const -> const std::vector<std::string>&;

private:
    struct HostBatch
    {
        int n { 0 };
        std::vector<float> images;
        std::vector<float> targets;
    };

    [[nodiscard]] auto load_sample(std::size_t sample_index, std::vector<float>& image_chw) const -> int;
    auto take_indices() -> std::vector<std::size_t>;
    [[nodiscard]] auto decode_indices(const std::vector<std::size_t>& indices) const -> HostBatch;
    auto upload_host_batch(HostBatch host, cudaStream_t stream) const -> Batch;
    auto launch_prefetch() -> void;
    auto join_prefetch() -> void;

    std::string split_root_;
    int batch_size_;
    int image_size_;
    bool shuffle_;
    std::vector<std::string> class_names_;
    std::vector<std::pair<std::string, int>> samples_;
    std::vector<std::size_t> order_;
    std::size_t cursor_ { 0 };
    std::mt19937 rng_;
    std::future<HostBatch> prefetch_;
};
