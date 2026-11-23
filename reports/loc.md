# Line counts

Counted with cloc 2.06. Third-party trees (`extern/`), build output, generated HTML, and `benchmarks/models/torch/` are not in this report. The LibTorch twin headers stay out so the count is the custom library, its tests, and its programs.

Files compiled as CUDA still use the `.cpp` extension, so cloc reports them as C++. The last table splits `src/` using `DEEPLEARN_CUDA_SOURCES` in `src/CMakeLists.txt`.

## Command

```text
cloc --by-file --exclude-dir=torch --exclude-ext=txt src include tests benchmarks scripts CMakeLists.txt
```

`benchmarks/models/torch` is excluded. Regenerate with `powershell -File scripts/loc.ps1`.

## By language

| Language | Files | Blank | Comment | Code |
| --- | --- | --- | --- | --- |
| C++ | 50 | 1445 | 76 | 8424 |
| C/C++ Header | 46 | 565 | 841 | 2927 |
| Python | 6 | 240 | 68 | 1542 |
| Bourne Shell | 4 | 81 | 31 | 582 |
| CMake | 4 | 70 | 28 | 479 |
| PowerShell | 3 | 12 | 5 | 99 |
| **Total** | 113 | 2413 | 1049 | 14053 |

## By directory

| Area | Files | Blank | Comment | Code |
| --- | --- | --- | --- | --- |
| Public headers | 22 | 348 | 751 | 1076 |
| Library sources | 16 | 635 | 64 | 4354 |
| Models | 7 | 64 | 102 | 222 |
| Benchmark and training programs | 31 | 333 | 0 | 3324 |
| Tests | 20 | 630 | 0 | 2375 |
| Scripts | 13 | 333 | 104 | 2223 |
| CMake | 4 | 70 | 28 | 479 |

## Library sources compiled by nvcc

These eight translation units are `LANGUAGE CUDA` in CMake. The host-only row is the remaining `src/**/*.cpp` files (loaders, flatten, pooling, logger, precision).

| Group | Files | Blank | Comment | Code |
| --- | --- | --- | --- | --- |
| nvcc translation units | 8 | 455 | 32 | 3252 |
| Host-only library sources | 8 | 180 | 32 | 1102 |

Comment lines are 7.5% of the code lines in the language table (1049 comments, 14053 code).
In the public headers the ratio is 69.8% (751 comments, 1076 code). That is where the Doxygen comments live.
Plain `.txt` files are excluded, so a thesis draft sitting under `scripts/` is not part of the count.
