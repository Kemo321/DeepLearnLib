/**
 * @file CSVLoader.hpp
 * @brief Load a whole numeric CSV into two host tensors.
 *
 * The training program, not this loader, splits rows into train and test.
 * There is no batching and no device upload here.
 */

#pragma once

#include "DeepLearnLib/core/Tensor.hpp"

#include <cstddef>
#include <string>

/**
 * @brief Feature matrix and target matrix read from one CSV file.
 *
 * The last @c target_columns of each row are targets. The preceding columns
 * are features. An optional header row is skipped. Both tensors live on the CPU.
 */
class CSVLoader
{
public:
    /**
     * @param csv_path Path to the file.
     * @param target_columns How many trailing columns are targets.
     * @param skip_header Ignore the first row.
     * @throws std::runtime_error if the file is missing or a row has the wrong width.
     */
    explicit CSVLoader(std::string csv_path, int target_columns = 1, bool skip_header = false);

    /** Feature matrix, shaped @c [rows, features], on the CPU. */
    [[nodiscard]] auto features() const -> const dl::Tensor&;

    /** Target matrix, shaped @c [rows, target_columns], on the CPU. */
    [[nodiscard]] auto targets() const -> const dl::Tensor&;

    /** Number of data rows. */
    [[nodiscard]] auto size() const -> std::size_t;

private:
    static auto from_parsed(const std::string& csv_path, int target_columns, bool skip_header) -> CSVLoader;
    CSVLoader(dl::Tensor features, dl::Tensor targets);

    dl::Tensor features_;
    dl::Tensor targets_;
};
