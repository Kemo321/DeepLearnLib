# Instalacja i budowanie

Instrukcja dotyczy repozytorium DeepLearnLib. Rdzeń biblioteki nie linkuje LibTorch. LibTorch jest opcjonalny i służy wyłącznie programom porównawczym (`*_torch`).

Skrót po angielsku jest w [README.md](README.md). Wygenerowany opis API jest jedną stroną HTML: <https://kemo321.github.io/DeepLearnLib/>.

## Wymagania

- karta NVIDIA i sterownik zgodny z wygenerowanym kodem SASS
- CMake 3.18 lub nowszy
- Ninja
- kompilator C++17
- CUDA Toolkit z nvcc, cuBLAS i cuDNN (w tym projekcie sprawdzano CUDA 12 oraz 13)
- OpenCV, jeśli mają powstać programy na zbiorach obrazowych

Bez OpenCV CMake buduje rdzeń i testy, które nie dekodują obrazów. Katalog `benchmarks/` jest wtedy pomijany.

Na Windowsie potrzebne są dodatkowo:

- Visual Studio z narzędziami C++ (MSVC). Skrypt szuka `vcvars64.bat`.
- Git Bash w `C:\Program Files\Git\bin\bash.exe`. Nie używać `bash.exe` z WSL: nvcc nie zobaczy nagłówków MSVC.

## Docker

Z katalogu repozytorium:

```bash
docker compose up -d --build
docker exec -it deeplearn_dev_container bash
./scripts/dev.sh
./scripts/menu.sh
```

Obraz bazuje na kontenerze NGC PyTorch, bo ta paczka ustala razem CUDA, cuDNN i Pythona. Biblioteka DeepLearnLib i tak nie zawiera LibTorch. Katalog repozytorium jest montowany w `/app`. `build/` i ccache są wolumenami. Kontener prosi o jedną kartę NVIDIA, a `/dev/shm` ma 32 GiB.

`./scripts/dev.sh` konfiguruje Ninja z `USE_CUDA=ON`, buduje `dllib_tests` i uruchamia testy.

## Windows

Z katalogu repozytorium, w PowerShell:

```powershell
powershell -File scripts\dev.ps1
```

Skrypt ładuje środowisko Visual C++ i dopiero wtedy uruchamia Git Bash oraz `scripts/dev.sh`. Samo `cmake` z zwykłego PowerShella kończy się błędem braku `stdlib.h`, bo nvcc kompiluje kod hosta kompilatorem MSVC.

cuDNN i OpenCV muszą być widoczne przy konfiguracji. W używanej konfiguracji lokalnej:

- `CUDNN_ROOT` wskazuje katalog cuDNN (nagłówki w `include`, biblioteki w `lib` albo `lib/x64`)
- `OpenCV_DIR` wskazuje katalog z `OpenCVConfig.cmake`

Jeśli cuDNN pochodzi z pakietu pip `nvidia-cudnn`, ustaw `CUDNN_ROOT` na ten katalog. CMake kopiuje znalezione DLL cuDNN obok `dllib_tests`, żeby loader nie wziął innej kopii z `PATH`.

## Budowanie ręczne

Linux, Git Bash albo kontener:

```bash
cmake -S . -B build -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel --target dllib_tests
./build/tests/dllib_tests
```

Konfiguracja odpytuje `nvidia-smi` i ustawia `CMAKE_CUDA_ARCHITECTURES` na `<major><minor>-real`. Dzięki temu sterownik nie kompiluje PTX przy pierwszym uruchomieniu. W CI, gdzie karty nie ma, lista architektur jest podana jawnie i sprawdzenie kończy się na linkowaniu.

Pomiary czasu rób w `Release`. `DEBUG_NUMERICS=ON` po każdym przebiegu warstwy sprawdza NaN i Inf i przy tym synchronizuje kartę, więc nie włącza się tego w przebiegu, z którego pochodzą liczby w pracy.

Pełne drzewo, łącznie z programami treningowymi, powstaje przez `cmake --build build --parallel` wtedy, gdy OpenCV zostało znalezione.

## Testy

Cel testów to `dllib_tests` (GoogleTest). Przypadki wymagające karty są pomijane, gdy `nvidia-smi` się nie powiedzie.

```bash
./build/tests/dllib_tests
```

Na Windowsie plik nazywa się `dllib_tests.exe`. `scripts/verify_gpu.ps1` uruchamia te testy dopiero po udanym `nvidia-smi` i odmawia pracy, gdy ustawione jest `GITHUB_ACTIONS`.

Pokrycie linii na kompilatorze GCC/Clang:

```bash
cmake -S . -B build_coverage -G Ninja -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug -DENABLE_COVERAGE=ON
cmake --build build_coverage --target dllib_tests
./build_coverage/tests/dllib_tests
lcov --capture --directory build_coverage --output-file coverage.info --ignore-errors gcov,source
lcov --extract coverage.info '*/src/*' '*/include/*' '*/tests/*' '*/benchmarks/models/*' '*/benchmarks/common/*' \
  --output-file coverage.filtered.info
python3 scripts/coverage_report.py --lcov coverage.filtered.info --output reports/coverage.md
```

`ENABLE_COVERAGE` dokłada `--coverage` tylko poza MSVC. Na Windowsie ten przełącznik nic nie zmienia, bo instrumentacja gcov nie obejmuje `cl.exe`.

## Dokumentacja z kodu

Komentarze Doxygen są w nagłówkach `include/DeepLearnLib/` oraz w `benchmarks/models/`. Wygeneruj HTML:

```bash
doxygen Doxyfile
```

Wynik jest w `docs/doxygen/html/index.html` i jest publikowany jako <https://kemo321.github.io/DeepLearnLib/>. Raport pokrycia jest pod <https://kemo321.github.io/DeepLearnLib/coverage.html>, a zestawienie linii pod <https://kemo321.github.io/DeepLearnLib/loc.html>. Po zmianie komentarzy wygeneruj HTML ponownie tym samym poleceniem.

## Metryki linii

Z zainstalowanym [cloc](https://github.com/AlDanial/cloc):

```bash
cloc src include tests benchmarks --exclude-dir=torch
```

Albo skrypt, który woła cloc i zapisuje `reports/loc.md`:

```powershell
powershell -File scripts\loc.ps1
```

Jądra CUDA leżą w plikach `.cpp` kompilowanych przez nvcc. cloc klasyfikuje je jako C++, bo rozszerzenie nie jest `.cu`. Raport rozdziela te pliki według listy w `src/CMakeLists.txt`.
