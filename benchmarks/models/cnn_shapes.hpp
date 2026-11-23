/**
 * @file cnn_shapes.hpp
 * @brief Spatial size of SimpleCNN after its two convolutions and two pools.
 *
 * Both the custom model and the LibTorch twin call @c pooled_cnn_features,
 * so the dense layer on each side sees the same number of inputs.
 */

#pragma once

/** Output side of one square convolution or pool. */
inline auto conv_spatial(int size, int kernel, int stride, int padding) -> int
{
    return ((size + (2 * padding) - kernel) / stride) + 1;
}

/**
 * @brief Features seen by the last dense layer of SimpleCNN.
 * @param image_size Square side before the first convolution.
 * @param channels Channels after the second convolution. The network uses 32.
 */
inline auto pooled_cnn_features(int image_size, int channels = 32) -> int
{
    int spatial = conv_spatial(image_size, 3, 1, 1);
    spatial = conv_spatial(spatial, 2, 2, 0);
    spatial = conv_spatial(spatial, 3, 1, 1);
    spatial = conv_spatial(spatial, 2, 2, 0);
    return channels * spatial * spatial;
}
