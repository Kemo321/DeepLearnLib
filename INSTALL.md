# Installation and build

These steps apply to the DeepLearnLib repository. The library core does not link LibTorch. LibTorch is optional and is used only by the comparison programs (`*_torch`).

A short overview is in [README.md](README.md). The generated API reference is one HTML site: <https://kemo321.github.io/DeepLearnLib/>.

## Requirements

- an NVIDIA GPU and a driver that matches the generated SASS
- CMake 3.18 or newer
- Ninja
- a C++17 compiler
- the CUDA Toolkit with nvcc, cuBLAS, and cuDNN (this project was checked with CUDA 12 and 13)
- OpenCV, if the image-dataset programs are to be built

Without OpenCV, CMake builds the core and the tests that do not decode images. The `benchmarks/` directory is then skipped.

Windows also needs:

- Visual Studio with the C++ tools (MSVC). The script looks for `vcvars64.bat`.
- Git Bash at `C:\Program Files\Git\bin\bash.exe`. Do not use `bash.exe` from WSL: nvcc will not see the MSVC headers.

## Docker

From the repository root:

```bash
docker compose up -d --build
docker exec -it deeplearn_dev_container bash
./scripts/dev.sh
./scripts/menu.sh
```

The image is based on the NGC PyTorch container, because that package pins CUDA, cuDNN, and Python together. DeepLearnLib still does not contain LibTorch. The repository directory is mounted at `/app`. `build/` and ccache are volumes. The container requests one NVIDIA GPU, and `/dev/shm` is 32 GiB.

`./scripts/dev.sh` configures Ninja with `USE_CUDA=ON`, builds `dllib_tests`, and runs the tests.

## Windows

From the repository root, in PowerShell:

```powershell
powershell -File scripts\dev.ps1
```

The script loads the Visual C++ environment and only then starts Git Bash and `scripts/dev.sh`. Running `cmake` from a plain PowerShell fails with a missing `stdlib.h`, because nvcc compiles the host code with the MSVC compiler.

cuDNN and OpenCV must be visible at configure time. In the local setup used here:

- `CUDNN_ROOT` points at the cuDNN directory (headers in `include`, libraries in `lib` or `lib/x64`)
- `OpenCV_DIR` points at the directory that contains `OpenCVConfig.cmake`

If cuDNN comes from the `nvidia-cudnn` pip package, set `CUDNN_ROOT` to that directory. CMake copies the cuDNN DLLs it finds next to `dllib_tests`, so the loader does not pick up another copy from `PATH`.

## Manual build

Linux, Git Bash, or the container:

```bash
cmake -S . -B build -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target dllib_tests
./build/tests/dllib_tests
```

Configuration queries `nvidia-smi` and sets `CMAKE_CUDA_ARCHITECTURES` to `<major><minor>-real`. The driver then does not compile PTX on the first launch. In CI, where there is no GPU, the architecture list is given explicitly and the check stops at link time.

Time measurements belong in `Release`. `DEBUG_NUMERICS=ON` checks for NaN and Inf after every layer pass and synchronizes the GPU to do so, so leave it off for the run that produces the numbers in the thesis.

The full tree, including the training programs, is produced by `cmake --build build --parallel` once OpenCV has been found.

## Tests

The test target is `dllib_tests` (GoogleTest). Cases that need a GPU are skipped when `nvidia-smi` fails.

```bash
./build/tests/dllib_tests
```

On Windows the file is named `dllib_tests.exe`. `scripts/verify_gpu.ps1` runs these tests only after `nvidia-smi` succeeds, and it refuses to run when `GITHUB_ACTIONS` is set.

Line coverage with GCC or Clang:

```bash
cmake -S . -B build_coverage -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
cmake --build build_coverage --target dllib_tests
./build_coverage/tests/dllib_tests
lcov --capture --directory build_coverage --output-file coverage.info --ignore-errors gcov,source
lcov --extract coverage.info '*/src/*' '*/include/*' '*/tests/*' '*/benchmarks/models/*' '*/benchmarks/common/*' \
  --output-file coverage.filtered.info
python3 scripts/coverage_report.py --lcov coverage.filtered.info --output reports/coverage.md
```

`ENABLE_COVERAGE` adds `--coverage` only outside MSVC. On Windows the switch changes nothing, because gcov instrumentation does not cover `cl.exe`.

## Documentation from the code

Doxygen comments are in the `include/DeepLearnLib/` headers and in `benchmarks/models/`. Generate the HTML:

```bash
doxygen Doxyfile
```

The result is `docs/doxygen/html/index.html` and is published at <https://kemo321.github.io/DeepLearnLib/>. The coverage report is at <https://kemo321.github.io/DeepLearnLib/coverage.html>, and the line-count table is at <https://kemo321.github.io/DeepLearnLib/loc.html>. After a comment change, generate the HTML again with the same command.

## Line metrics

With [cloc](https://github.com/AlDanial/cloc) installed:

```bash
cloc src include tests benchmarks --exclude-dir=torch
```

Or the script that calls cloc and writes `reports/loc.md`:

```powershell
powershell -File scripts\loc.ps1
```

CUDA kernels live in `.cpp` files compiled by nvcc. cloc classifies them as C++, because the extension is not `.cu`. The report separates those files using the list in `src/CMakeLists.txt`.
