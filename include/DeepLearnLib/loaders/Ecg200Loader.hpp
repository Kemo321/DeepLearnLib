/**
 * @file Ecg200Loader.hpp
 * @brief Reader for the UCR ECG200 text files.
 *
 * Each data line is a series of equal length followed by a class label.
 * The loader does not build the @c [batch, time, 1] tensor. The training
 * program does that from @c values and @c labels.
 */

#pragma once

#include <filesystem>
#include <string>
#include <vector>

/**
 * @brief One ECG200 split loaded into host memory.
 *
 * @c values is row-major, one series of @c length samples per label.
 * Labels are class indices. @c class_names lists those classes in index order.
 */
struct Ecg200Data
{
    /** Samples in every series. Zero when the file has no rows. */
    int length = 0;

    /** Concatenated series, @c size() * length floats. */
    std::vector<float> values;

    /** Class index of each series. */
    std::vector<int> labels;

    /** Names parallel to the class indices. */
    std::vector<std::string> class_names;

    /** Number of series. */
    [[nodiscard]] auto size() const -> int
    {
        return static_cast<int>(labels.size());
    }
};

/**
 * @brief Parse one ECG200 file.
 * @param path Text file, either the raw UCR layout or the archive copy used in tests.
 * @throws std::runtime_error if the path cannot be read or a row is malformed.
 */
[[nodiscard]] auto load_ecg200_file(const std::filesystem::path& path) -> Ecg200Data;

/**
 * @brief Resolve the train or test file under an ECG200 root.
 * @param root Directory that contains the split files.
 * @param which @c "train" or @c "test".
 * @return Path of the split file.
 * @throws std::runtime_error if the split is not present.
 */
[[nodiscard]] auto find_ecg200_split(const std::filesystem::path& root, const std::string& which)
    -> std::filesystem::path;
