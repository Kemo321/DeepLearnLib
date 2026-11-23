# Coverage

This report maps library sources, application models, and the shared benchmark headers onto the GoogleTest suite.
A file counts as covered when a test names it. Training and forward programs are drivers: they are not part of that percentage, because the loop, the CSV, and the models they call are what the tests check.

Line, function, and branch counts are from the lcov artifact of the Debug CI job on commit 6f8e225 (17 Oct 2026, lcov 1.15). The runner has no GPU. nvcc did not emit gcov records, so the CUDA translation units are absent from the trace rather than scored as zero. Totals count library, model, and shared-header records only, not libstdc++ or the CUDA headers.

- Files named by a test: **49 / 66** (74.2%)
- Execution on 23 Nov 2026: `build/tests/dllib_tests.exe` passed **147 / 147** tests in 1.4 s. None were skipped. The executable is the Release build from 17 Oct 2026, 20:16. Later edits on those trees are comments and blank lines, so they do not change that program.
- Lines hit (lcov): **375 / 976** (38.4%)
- Branches hit (lcov): **347 / 1596** (21.7%)
- Functions hit (lcov): **49 / 142** (34.5%)

## Library, models, and shared headers

Code is the cloc code-line count. Hits and branches come from lcov when `--lcov` is passed.

| File | Tests | Code | Lines | Functions | Branches |
| --- | --- | --- | --- | --- | --- |
| `benchmarks/common/classification_eval.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 241 | 19 / 19 | 2 / 2 | 8 / 8 |
| `benchmarks/common/classification_vis.hpp` | — | 52 | — | — | — |
| `benchmarks/common/custom_step.hpp` | — | 29 | — | — | — |
| `benchmarks/common/experiment_config.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 179 | 45 / 54 | 5 / 5 | 66 / 144 |
| `benchmarks/common/forward_plan.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 67 | 31 / 33 | 2 / 2 | 42 / 90 |
| `benchmarks/common/image_train_custom.hpp` | — | 91 | — | — | — |
| `benchmarks/common/image_train_torch.hpp` | — | 143 | — | — | — |
| `benchmarks/common/indexed_train_custom.hpp` | — | 110 | — | — | — |
| `benchmarks/common/indexed_train_torch.hpp` | — | 128 | — | — | — |
| `benchmarks/common/prefetch_batch.hpp` | — | 46 | — | — | — |
| `benchmarks/common/row_batch.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 121 | 41 / 47 | 4 / 4 | 35 / 68 |
| `benchmarks/common/run_metrics.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 58 | 26 / 30 | 5 / 6 | 35 / 80 |
| `benchmarks/common/split_metrics.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 35 | 15 / 15 | 3 / 3 | 6 / 6 |
| `benchmarks/common/tabular_common.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 63 | 7 / 7 | 1 / 1 | 3 / 4 |
| `benchmarks/common/tabular_pipeline.hpp` | — | 55 | — | — | — |
| `benchmarks/common/torch_optim.hpp` | — | 56 | — | — | — |
| `benchmarks/common/torch_paired_init.hpp` | — | 116 | — | — | — |
| `benchmarks/common/train_loop.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 17 | 9 / 9 | 1 / 1 | 9 / 18 |
| `benchmarks/models/cnn_shapes.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 13 | 8 / 8 | 2 / 2 | — |
| `benchmarks/models/GruClassifier.cpp` | `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | 43 | 7 / 25 | 1 / 6 | 6 / 30 |
| `benchmarks/models/GruClassifier.hpp` | `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | 20 | — | — | — |
| `benchmarks/models/SimpleCNN.cpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp` | 62 | 0 / 36 | 0 / 6 | 0 / 60 |
| `benchmarks/models/SimpleCNN.hpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp` | 20 | — | — | — |
| `benchmarks/models/TabularMLP.cpp` | `tests/models/test_model_parity.cpp`, `tests/models/test_tabular_mlp.cpp` | 44 | 7 / 26 | 1 / 6 | 6 / 34 |
| `benchmarks/models/TabularMLP.hpp` | `tests/models/test_model_parity.cpp`, `tests/models/test_tabular_mlp.cpp` | 20 | — | — | — |
| `benchmarks/models/torch/TorchGruClassifier.hpp` | `tests/models/test_model_parity.cpp` | — | — | — | — |
| `benchmarks/models/torch/TorchSimpleCNN.hpp` | `tests/models/test_model_parity.cpp` | — | — | — | — |
| `benchmarks/models/torch/TorchTabularMLP.hpp` | `tests/models/test_model_parity.cpp` | — | — | — | — |
| `include/DeepLearnLib/core/Batch.hpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/loaders/test_classification_loader.cpp`, `tests/loaders/test_packed_image_loader.cpp`, `tests/losses/test_generic.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp` | 7 | — | — | — |
| `include/DeepLearnLib/core/Layer.hpp` | `tests/core/test_network.cpp`, `tests/layers/test_activations.cpp`, `tests/layers/test_conv2d.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_gru.cpp`, `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/losses/test_numerics.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp` | 58 | 1 / 18 | 1 / 9 | — |
| `include/DeepLearnLib/core/Network.hpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/losses/test_numerics.cpp` | 26 | — | — | — |
| `include/DeepLearnLib/core/Precision.hpp` | — | 14 | — | — | — |
| `include/DeepLearnLib/core/Tensor.hpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_activations.cpp`, `tests/layers/test_conv2d.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_gru.cpp`, `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/loaders/test_classification_loader.cpp`, `tests/loaders/test_packed_image_loader.cpp`, `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp`, `tests/test_helpers.hpp` | 242 | 5 / 19 | 2 / 6 | 1 / 26 |
| `include/DeepLearnLib/layers/Conv2d.hpp` | `tests/layers/test_conv2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | 180 | 0 / 13 | 0 / 3 | 0 / 22 |
| `include/DeepLearnLib/layers/Flatten.hpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_flatten.cpp` | 14 | 0 / 1 | 0 / 1 | — |
| `include/DeepLearnLib/layers/FullyConnected.hpp` | `tests/core/test_network.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/losses/test_numerics.cpp` | 41 | — | — | — |
| `include/DeepLearnLib/layers/Gru.hpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/layers/test_gru.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | 35 | — | — | — |
| `include/DeepLearnLib/layers/LeakyReLU.hpp` | `tests/layers/test_activations.cpp` | 18 | — | — | — |
| `include/DeepLearnLib/layers/MaxPool2d.hpp` | `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | 46 | — | — | — |
| `include/DeepLearnLib/loaders/ClassificationLoader.hpp` | `tests/loaders/test_classification_loader.cpp` | 50 | — | — | — |
| `include/DeepLearnLib/loaders/CSVLoader.hpp` | `tests/losses/test_generic.cpp` | 17 | — | — | — |
| `include/DeepLearnLib/loaders/Ecg200Loader.hpp` | `tests/layers/test_gru.cpp` | 18 | 2 / 2 | 1 / 1 | — |
| `include/DeepLearnLib/loaders/PackedImageLoader.hpp` | `tests/loaders/test_packed_image_loader.cpp` | 56 | — | — | — |
| `include/DeepLearnLib/losses/Losses.hpp` | `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp` | 8 | — | — | — |
| `include/DeepLearnLib/utils/ExperimentSeed.hpp` | `tests/benchmarks/test_train_metrics.cpp` | 60 | 21 / 22 | 3 / 4 | 4 / 6 |
| `include/DeepLearnLib/utils/Logger.hpp` | — | 22 | — | — | — |
| `include/DeepLearnLib/utils/Nvtx.hpp` | — | 52 | 0 / 6 | 0 / 2 | — |
| `include/DeepLearnLib/utils/ParallelFor.hpp` | `tests/loaders/test_classification_loader.cpp` | 66 | 0 / 30 | 0 / 3 | 0 / 32 |
| `include/DeepLearnLib/utils/Profiler.hpp` | — | 22 | — | — | — |
| `include/DeepLearnLib/utils/SafeMath.hpp` | `tests/losses/test_numerics.cpp` | 24 | 0 / 4 | 0 / 2 | — |
| `src/core/Network.cpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/losses/test_numerics.cpp` | 221 | — | — | — |
| `src/core/Precision.cpp` | — | 13 | 0 / 5 | 0 / 1 | 0 / 2 |
| `src/core/Tensor.cpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_activations.cpp`, `tests/layers/test_conv2d.cpp`, `tests/layers/test_flatten.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_gru.cpp`, `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/loaders/test_classification_loader.cpp`, `tests/loaders/test_packed_image_loader.cpp`, `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp`, `tests/models/test_simplecnn.cpp`, `tests/models/test_tabular_mlp.cpp`, `tests/test_helpers.hpp` | 1200 | — | — | — |
| `src/layers/Conv2d.cpp` | `tests/layers/test_conv2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | 641 | — | — | — |
| `src/layers/Flatten.cpp` | `tests/core/test_network.cpp`, `tests/core/test_tensor.cpp`, `tests/layers/test_flatten.cpp` | 47 | 0 / 25 | 0 / 3 | 0 / 54 |
| `src/layers/FullyConnected.cpp` | `tests/core/test_network.cpp`, `tests/layers/test_fullyconnected.cpp`, `tests/layers/test_torch_equivalence.cpp`, `tests/losses/test_numerics.cpp` | 228 | — | — | — |
| `src/layers/Gru.cpp` | `tests/benchmarks/test_train_metrics.cpp`, `tests/layers/test_gru.cpp`, `tests/models/test_gru_classifier.cpp`, `tests/models/test_model_parity.cpp` | 630 | — | — | — |
| `src/layers/LeakyReLU.cpp` | `tests/layers/test_activations.cpp` | 112 | — | — | — |
| `src/layers/MaxPool2d.cpp` | `tests/layers/test_maxpool2d.cpp`, `tests/layers/test_torch_equivalence.cpp` | 135 | 0 / 86 | 0 / 11 | 0 / 162 |
| `src/loaders/ClassificationLoader.cpp` | `tests/loaders/test_classification_loader.cpp` | 270 | — | — | — |
| `src/loaders/CSVLoader.cpp` | `tests/losses/test_generic.cpp` | 136 | 10 / 73 | 3 / 7 | 7 / 122 |
| `src/loaders/Ecg200Loader.cpp` | `tests/layers/test_gru.cpp` | 185 | 87 / 110 | 6 / 7 | 95 / 230 |
| `src/loaders/PackedImageLoader.cpp` | `tests/loaders/test_packed_image_loader.cpp` | 251 | 0 / 155 | 0 / 23 | 0 / 158 |
| `src/losses/Losses.cpp` | `tests/losses/test_generic.cpp`, `tests/losses/test_numerics.cpp` | 153 | 1 / 63 | 1 / 10 | 0 / 188 |
| `src/utils/Logger.cpp` | — | 65 | 33 / 35 | 5 / 5 | 24 / 52 |
| `src/utils/Profiler.cpp` | — | 67 | — | — | — |

## Lowest line coverage

The ten traced files with the smallest share of hit lines. A zero means the record was instrumented and no test executed it. CI has no GPU, so device-only lines stay at zero there.

| File | Lines | Branches |
| --- | --- | --- |
| `benchmarks/models/SimpleCNN.cpp` | 0 / 36 | 0 / 60 |
| `include/DeepLearnLib/layers/Conv2d.hpp` | 0 / 13 | 0 / 22 |
| `include/DeepLearnLib/layers/Flatten.hpp` | 0 / 1 | — |
| `include/DeepLearnLib/utils/Nvtx.hpp` | 0 / 6 | — |
| `include/DeepLearnLib/utils/ParallelFor.hpp` | 0 / 30 | 0 / 32 |
| `include/DeepLearnLib/utils/SafeMath.hpp` | 0 / 4 | — |
| `src/core/Precision.cpp` | 0 / 5 | 0 / 2 |
| `src/layers/Flatten.cpp` | 0 / 25 | 0 / 54 |
| `src/layers/MaxPool2d.cpp` | 0 / 86 | 0 / 162 |
| `src/loaders/PackedImageLoader.cpp` | 0 / 155 | 0 / 158 |

## Files with no matching test

- `benchmarks/common/classification_vis.hpp`
- `benchmarks/common/custom_step.hpp`
- `benchmarks/common/image_train_custom.hpp`
- `benchmarks/common/image_train_torch.hpp`
- `benchmarks/common/indexed_train_custom.hpp`
- `benchmarks/common/indexed_train_torch.hpp`
- `benchmarks/common/prefetch_batch.hpp`
- `benchmarks/common/tabular_pipeline.hpp`
- `benchmarks/common/torch_optim.hpp`
- `benchmarks/common/torch_paired_init.hpp`
- `include/DeepLearnLib/core/Precision.hpp`
- `include/DeepLearnLib/utils/Logger.hpp`
- `include/DeepLearnLib/utils/Nvtx.hpp`
- `include/DeepLearnLib/utils/Profiler.hpp`
- `src/core/Precision.cpp`
- `src/utils/Logger.cpp`
- `src/utils/Profiler.cpp`

## Training and benchmark drivers

These programs are the paired custom and Torch entry points. They are compiled by the release job and are not unit tests.

- `benchmarks/forward/bench_forward_custom.cpp` — paired binary
- `benchmarks/forward/bench_forward_torch.cpp` — paired binary
- `benchmarks/micro/bench_micro_common.hpp` — paired binary
- `benchmarks/micro/bench_micro_more.cpp` — paired binary
- `benchmarks/micro/bench_micro_ops.cpp` — paired binary
- `benchmarks/train/train_cifar_custom.cpp` — paired binary
- `benchmarks/train/train_cifar_torch.cpp` — paired binary
- `benchmarks/train/train_gru_custom.cpp` — paired binary
- `benchmarks/train/train_gru_torch.cpp` — paired binary
- `benchmarks/train/train_mnist_custom.cpp` — paired binary
- `benchmarks/train/train_mnist_torch.cpp` — paired binary
- `benchmarks/train/train_tabular_custom.cpp` — paired binary
- `benchmarks/train/train_tabular_torch.cpp` — paired binary

## Test cases

GoogleTest cases in `tests/`: **147**.
The file table above is a name match, not a line trace. A line percentage comes only from an lcov tracefile.

| Test file | Suite | Case |
| --- | --- | --- |
| `tests/benchmarks/test_train_metrics.cpp` | `SplitMetrics` | `WeightsLossAndAccuracyBySampleCount` |
| `tests/benchmarks/test_train_metrics.cpp` | `SplitMetrics` | `EmptySplitIsZero` |
| `tests/benchmarks/test_train_metrics.cpp` | `TrainTestSplit` | `SameSeedIsSamePartition` |
| `tests/benchmarks/test_train_metrics.cpp` | `TrainTestSplit` | `RejectsASingleRow` |
| `tests/benchmarks/test_train_metrics.cpp` | `HostRows` | `ShapeTailMustMatchWidth` |
| `tests/benchmarks/test_train_metrics.cpp` | `MetricsCsv` | `KeepsSubSecondTime` |
| `tests/benchmarks/test_train_metrics.cpp` | `SupervisedEpochs` | `WritesOneWeightedRowPerEpoch` |
| `tests/benchmarks/test_train_metrics.cpp` | `ForwardPlan` | `BothStacksReadTheSameCases` |
| `tests/benchmarks/test_train_metrics.cpp` | `ExperimentSeed` | `ShuffleRepeatsForTheSameSeed` |
| `tests/benchmarks/test_train_metrics.cpp` | `LearningRateSchedule` | `DropsOnTheConfiguredEpochs` |
| `tests/benchmarks/test_train_metrics.cpp` | `CnnShapes` | `TwoPoolsDivideTheSideByFour` |
| `tests/benchmarks/test_train_metrics.cpp` | `LabelEncoding` | `OneHotAndArgmaxAgree` |
| `tests/core/test_network.cpp` | `NetworkTest` | `NullLayerThrows` |
| `tests/core/test_network.cpp` | `NetworkTest` | `ForwardThroughFlattenAndDense` |
| `tests/core/test_network.cpp` | `NetworkTest` | `SaveLoadRoundTripRestoresWeights` |
| `tests/core/test_network.cpp` | `NetworkTest` | `LoadMissingFileThrows` |
| `tests/core/test_network.cpp` | `NetworkTest` | `LoadLayerCountMismatchThrows` |
| `tests/core/test_network.cpp` | `NetworkTest` | `AssignsLearningRateToLayers` |
| `tests/core/test_tensor.cpp` | `TensorConstructorTest` | `CpuAllocationAndZeroInitialization` |
| `tests/core/test_tensor.cpp` | `TensorConstructorTest` | `ScalarCpuAllocation` |
| `tests/core/test_tensor.cpp` | `TensorConstructorTest` | `GpuAllocationBehavior` |
| `tests/core/test_tensor.cpp` | `TensorConstructorTest` | `ViewConstructorSharesMemoryOwnership` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `OneDimensionalAllocationShapeAndStrides` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `TwoDimensionalAllocationShapeAndStrides` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ThreeDimensionalAllocationShapeAndStrides` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `NchwStridesMatchNetworkLayout` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `FromHostVectorRoundTrip` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `FromHostPointerRoundTrip` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `FromHostCpuDeviceRoundTrip` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `FromHostRejectsMismatchedBufferSize` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `FromHostRejectsNullPointer` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ZerosLikeMatchesShapeDeviceAndValues` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `SquareMatmul` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `RectangularMatmul` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `LogicalTransposeMatmulMatchesPhysicalTranspose` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `LogicalTransposeBMatmulMatchesPhysicalTranspose` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `InPlaceAddMulAndAddScaledDoNotAllocateTemps` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `MatmulIntoAccumulatesIntoExistingBuffer` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `AddRowBroadcastsBiasAcrossBatch` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `AddSumRowsAccumulatesWithBeta` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `IdentityMatmulLeavesMatrixUnchanged` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `MatmulMismatchedInnerDimensionsThrows` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ElementwiseTensorAddition` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ElementwiseTensorSubtraction` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ElementwiseTensorMultiplication` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `TensorScalarMultiplicationAndAddition` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ElementwiseSizeMismatchThrows` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ElementwiseCpuOperandThrows` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ClampBoundsValues` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ClampInvalidRangeThrows` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `GlobalSumPositiveValues` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `GlobalSumNegativeAndMixedValues` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `SumAlongAxisThrows` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ViewReshapeSharesStorageAndValues` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ViewInfersMinusOneDimension` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ViewIncompatibleShapeThrows` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `TransposeTwoDimensional` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `TransposeThenMatmulMatchesOriginalProductLayout` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `TransposeRejectsNonTwoDimensionalTensors` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `VectorMatrixMatmul` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `MatmulOnCpuThrows` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ChainedElementwiseOps` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ClampEqualBoundsSaturates` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ViewFlattensWithMinusOne` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ViewRejectsInvalidNegativeDimension` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ViewIdentityKeepsShape` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `SumOfZerosIsZero` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `TransposeOneByN` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `MoveConstructorTransfersOwnership` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `ZerosLikeCpuSourceStaysOnCpu` |
| `tests/core/test_tensor.cpp` | `GpuTensorTest` | `CpuToHostRoundTrip` |
| `tests/layers/test_activations.cpp` | `ActivationTest` | `LeakyReLUForwardAndBackwardSlope` |
| `tests/layers/test_activations.cpp` | `ActivationTest` | `LeakyReLUDefaultSlopeOnNegatives` |
| `tests/layers/test_activations.cpp` | `ActivationTest` | `LeakyReLUZeroSlopeIsRelu` |
| `tests/layers/test_activations.cpp` | `ActivationTest` | `LeakyReLURejectsCpuAndMismatchedBackward` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `ForwardOutputShape` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `OnesKernelMatchesWindowSum` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `IdentityOneByOneAndBackward` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `BackwardWithoutForwardThrows` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `InvalidConstructorArgumentsThrow` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `ForwardRejectsChannelMismatchAndCpuInput` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `StrideTwoHalvesSpatialSize` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `ZeroWeightsAddBias` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `ParameterRoundTripAndGpuTo` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `StepUpdatesWeightsAfterBackward` |
| `tests/layers/test_conv2d.cpp` | `Conv2dTest` | `BackwardRejectsMismatchedGradientShape` |
| `tests/layers/test_flatten.cpp` | `FlattenTest` | `ReshapesAndUnflattensGradient` |
| `tests/layers/test_flatten.cpp` | `FlattenTest` | `AlreadyRankTwoIsUnchanged` |
| `tests/layers/test_flatten.cpp` | `FlattenTest` | `FiveDimensionalInput` |
| `tests/layers/test_flatten.cpp` | `FlattenTest` | `ScalarAndCpuInputsThrow` |
| `tests/layers/test_flatten.cpp` | `FlattenTest` | `BackwardWithoutForwardThrows` |
| `tests/layers/test_flatten.cpp` | `FlattenTest` | `OneDimensionalTreatsLengthAsBatch` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `ForwardMatchesMatmulPlusBias` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `BackwardAndStepUpdateWeights` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `InvalidSizesThrow` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `ForwardRejectsWrongRankDeviceAndFeatures` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `BackwardWithoutForwardThrows` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `SecondBackwardWithoutForwardThrows` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `BackwardRejectsBatchMismatch` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `ZeroWeightsReturnBias` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `ParameterRoundTripAndGpuTo` |
| `tests/layers/test_fullyconnected.cpp` | `FullyConnectedTest` | `DefaultInitializationIsFinite` |
| `tests/layers/test_gru.cpp` | `Ecg200Loader` | `ReadsTextAndTsLabels` |
| `tests/layers/test_gru.cpp` | `Ecg200Loader` | `ReadsArchiveWhenPresent` |
| `tests/layers/test_gru.cpp` | `GruShape` | `RejectsEmptySizes` |
| `tests/layers/test_gru.cpp` | `GruGpuTest` | `FinalStateShapeAndWeightGradient` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `DownsamplesAndRoutesGradientToArgmax` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `FourByFourWindows` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `InvalidConstructorArgumentsThrow` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `ForwardRejectsCpuAndNonNchw` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `BackwardWithoutForwardThrows` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `BackwardRejectsMismatchedGradientShape` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `MultiChannelPoolsIndependently` |
| `tests/layers/test_maxpool2d.cpp` | `MaxPool2dTest` | `RepeatedForwardReusesDescriptors` |
| `tests/layers/test_torch_equivalence.cpp` | `TorchEquivalenceTest` | `Conv2dForwardAndBackwardMatchLibTorch` |
| `tests/layers/test_torch_equivalence.cpp` | `TorchEquivalenceTest` | `MaxPool2dForwardAndBackwardMatchLibTorch` |
| `tests/layers/test_torch_equivalence.cpp` | `TorchEquivalenceTest` | `FullyConnectedForwardAndBackwardMatchLibTorch` |
| `tests/loaders/test_classification_loader.cpp` | `ClassificationLoaderTest` | `DiscoversClassesAndYieldsNchwOneHotBatches` |
| `tests/loaders/test_classification_loader.cpp` | `ClassificationLoaderTest` | `CoversEverySampleAcrossPrefetchedBatchesWithoutOverlap` |
| `tests/loaders/test_classification_loader.cpp` | `ClassificationLoaderTest` | `ResetRestartsEpochAndAllowsReuse` |
| `tests/loaders/test_classification_loader.cpp` | `ClassificationLoaderTest` | `LockedClassNamesKeepOneHotWidthWhenAFolderIsMissing` |
| `tests/loaders/test_classification_loader.cpp` | `ClassificationLoaderTest` | `RejectsInvalidConstructorArguments` |
| `tests/loaders/test_classification_loader.cpp` | `ParallelForCpuTest` | `ExecutesEveryIndexExactlyOnce` |
| `tests/loaders/test_classification_loader.cpp` | `ParallelForCpuTest` | `PropagatesWorkerExceptions` |
| `tests/loaders/test_packed_image_loader.cpp` | `PackedImageLoaderTest` | `YieldsNchwFloatAndOneHotLabels` |
| `tests/loaders/test_packed_image_loader.cpp` | `PackedImageLoaderTest` | `RejectsBadMagic` |
| `tests/losses/test_generic.cpp` | `GenericComponentTest` | `CrossEntropyPrefersTheCorrectOneHotClass` |
| `tests/losses/test_generic.cpp` | `GenericComponentTest` | `CrossEntropyGradientIsSoftmaxMinusTargetOverBatch` |
| `tests/losses/test_generic.cpp` | `GenericComponentTest` | `CsvLoaderReadsFeaturesAndTargets` |
| `tests/losses/test_generic.cpp` | `CsvLoaderCpuTest` | `MissingFileThrows` |
| `tests/losses/test_numerics.cpp` | `GpuNumericsTest` | `CrossEntropyStaysFiniteForOverflowingLogits` |
| `tests/losses/test_numerics.cpp` | `GpuNumericsTest` | `CrossEntropyClampsNearZeroProbabilities` |
| `tests/losses/test_numerics.cpp` | `GpuNumericsTest` | `NetworkStoresConfigurableGradientClip` |
| `tests/losses/test_numerics.cpp` | `GpuNumericsTest` | `ClipLossGradientBoundsEveryElement` |
| `tests/losses/test_numerics.cpp` | `GpuNumericsTest` | `ParameterGradientClipBoundsDenseUpdates` |
| `tests/losses/test_numerics.cpp` | `GpuNumericsTest` | `HasNonFiniteDetectsInfAndNan` |
| `tests/losses/test_numerics.cpp` | `GpuNumericsTest` | `DefaultNetworkGradientClipIsDisabled` |
| `tests/models/test_gru_classifier.cpp` | `GruClassifierTest` | `FinalHiddenFeedsTheClassHead` |
| `tests/models/test_gru_classifier.cpp` | `GruClassifierShape` | `RejectsNonPositiveSizes` |
| `tests/models/test_model_parity.cpp` | `ModelParityTest` | `SimpleCnnParameterCountMatchesTorch` |
| `tests/models/test_model_parity.cpp` | `ModelParityTest` | `TabularMlpParameterCountMatchesTorch` |
| `tests/models/test_model_parity.cpp` | `ModelParityTest` | `GruClassifierOutputShapeMatchesTorch` |
| `tests/models/test_simplecnn.cpp` | `SimpleCnnTest` | `ForwardLogitsMatchClassCount` |
| `tests/models/test_simplecnn.cpp` | `SimpleCnnTest` | `TrainableStackHasExpectedDepth` |
| `tests/models/test_simplecnn.cpp` | `SimpleCnnTest` | `AcceptsSingleChannelMnistInput` |
| `tests/models/test_tabular_mlp.cpp` | `TabularMlpTest` | `LogitsMatchClassCount` |
| `tests/models/test_tabular_mlp.cpp` | `TabularMlp` | `RejectsNonPositiveSizes` |
