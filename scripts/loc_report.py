#!/usr/bin/env python3
"""Write reports/loc.md from cloc.

CUDA kernels in this repository live in .cpp files compiled by nvcc, so cloc
labels them C++. The CUDA table below is that same count, split with the
DEEPLEARN_CUDA_SOURCES list from src/CMakeLists.txt.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "loc.md"

CUDA_SOURCES = {
    "src/core/Tensor.cpp",
    "src/layers/Conv2d.cpp",
    "src/layers/LeakyReLU.cpp",
    "src/layers/FullyConnected.cpp",
    "src/layers/Gru.cpp",
    "src/core/Network.cpp",
    "src/utils/Profiler.cpp",
    "src/losses/Losses.cpp",
}

AREAS = (
    ("Public headers", "include/"),
    ("Library sources", "src/"),
    ("Tests", "tests/"),
    ("Models", "benchmarks/models/"),
    ("Benchmark and training programs", "benchmarks/"),
    ("Scripts", "scripts/"),
    ("CMake", "CMakeLists.txt"),
)


def run_cloc(cloc: str) -> dict:
    command = [
        cloc,
        "--json",
        "--by-file",
        "--quiet",
        "--exclude-dir=torch",
        "--exclude-ext=txt",
        "src",
        "include",
        "tests",
        "benchmarks",
        "scripts",
        "CMakeLists.txt",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def area_of(path: str) -> str:
    posix = path.replace("\\", "/")
    if posix.endswith("CMakeLists.txt"):
        return "CMake"
    if posix.startswith("benchmarks/models/"):
        return "Models"
    if posix.startswith("benchmarks/"):
        return "Benchmark and training programs"
    if posix.startswith("include/"):
        return "Public headers"
    if posix.startswith("src/"):
        return "Library sources"
    if posix.startswith("tests/"):
        return "Tests"
    if posix.startswith("scripts/"):
        return "Scripts"
    return "Other"


def blank_row() -> dict[str, int]:
    return {"files": 0, "blank": 0, "comment": 0, "code": 0}


def add(bucket: dict[str, int], row: dict) -> None:
    bucket["files"] += 1
    bucket["blank"] += int(row.get("blank", 0))
    bucket["comment"] += int(row.get("comment", 0))
    bucket["code"] += int(row.get("code", 0))


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Write reports/loc.md using cloc.")
    parser.add_argument("--cloc", required=True, help="Path to the cloc executable.")
    args = parser.parse_args()

    payload = run_cloc(args.cloc)
    header = payload.get("header", {})
    version = str(header.get("cloc_version", "unknown"))

    by_lang: dict[str, dict[str, int]] = defaultdict(blank_row)
    by_area: dict[str, dict[str, int]] = defaultdict(blank_row)
    cuda = blank_row()
    host_src = blank_row()
    by_file: dict[str, int] = {}

    for key, row in payload.items():
        if key in {"header", "SUM"} or not isinstance(row, dict) or "code" not in row:
            continue
        path = key.replace("\\", "/")
        by_file[path] = int(row["code"])
        language = str(row.get("language", "Unknown"))
        add(by_lang[language], row)
        add(by_area[area_of(path)], row)
        if path in CUDA_SOURCES:
            add(cuda, row)
        elif path.startswith("src/") and path.endswith(".cpp"):
            add(host_src, row)

    code_total = sum(item["code"] for item in by_lang.values())
    comment_total = sum(item["comment"] for item in by_lang.values())
    blank_total = sum(item["blank"] for item in by_lang.values())
    file_total = sum(item["files"] for item in by_lang.values())

    order = [
        "Public headers",
        "Library sources",
        "Models",
        "Benchmark and training programs",
        "Tests",
        "Scripts",
        "CMake",
        "Other",
    ]

    lines = [
        "# Line counts",
        "",
        f"Counted with cloc {version}. Third-party trees (`extern/`), build output, generated HTML, and `benchmarks/models/torch/` are not in this report. The LibTorch twin headers stay out so the count is the custom library, its tests, and its programs.",
        "",
        "Files compiled as CUDA still use the `.cpp` extension, so cloc reports them as C++. The last table splits `src/` using `DEEPLEARN_CUDA_SOURCES` in `src/CMakeLists.txt`.",
        "",
        "## Command",
        "",
        "```text",
        "cloc --by-file --exclude-dir=torch --exclude-ext=txt src include tests benchmarks scripts CMakeLists.txt",
        "```",
        "",
        "`benchmarks/models/torch` is excluded. Regenerate with `powershell -File scripts/loc.ps1`.",
        "",
        "## By language",
        "",
    ]
    lang_rows = []
    for language, item in sorted(by_lang.items(), key=lambda pair: -pair[1]["code"]):
        lang_rows.append(
            [language, str(item["files"]), str(item["blank"]), str(item["comment"]), str(item["code"])]
        )
    lang_rows.append(["**Total**", str(file_total), str(blank_total), str(comment_total), str(code_total)])
    lines.extend(table(["Language", "Files", "Blank", "Comment", "Code"], lang_rows))
    lines.extend(["", "## By directory", ""])
    area_rows = []
    for name in order:
        item = by_area.get(name)
        if not item or item["files"] == 0:
            continue
        area_rows.append([name, str(item["files"]), str(item["blank"]), str(item["comment"]), str(item["code"])])
    lines.extend(table(["Area", "Files", "Blank", "Comment", "Code"], area_rows))

    ratio = (100.0 * comment_total / code_total) if code_total else 0.0
    lines.extend(
        [
            "",
            "## Library sources compiled by nvcc",
            "",
            "These eight translation units are `LANGUAGE CUDA` in CMake. The host-only row is the remaining `src/**/*.cpp` files (loaders, flatten, pooling, logger, precision).",
            "",
        ]
    )
    lines.extend(
        table(
            ["Group", "Files", "Blank", "Comment", "Code"],
            [
                ["nvcc translation units", str(cuda["files"]), str(cuda["blank"]), str(cuda["comment"]), str(cuda["code"])],
                [
                    "Host-only library sources",
                    str(host_src["files"]),
                    str(host_src["blank"]),
                    str(host_src["comment"]),
                    str(host_src["code"]),
                ],
            ],
        )
    )
    header_item = by_area.get("Public headers", blank_row())
    header_ratio = (100.0 * header_item["comment"] / header_item["code"]) if header_item["code"] else 0.0
    lines.extend(
        [
            "",
            f"Comment lines are {ratio:.1f}% of the code lines in the language table ({comment_total} comments, {code_total} code).",
            f"In the public headers the ratio is {header_ratio:.1f}% ({header_item['comment']} comments, {header_item['code']} code). That is where the Doxygen comments live.",
            "Plain `.txt` files are excluded, so a thesis draft sitting under `scripts/` is not part of the count.",
            "",
        ]
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    (REPORT.parent / "loc-by-file.json").write_text(json.dumps(by_file, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT).as_posix()} ({code_total} code lines, cloc {version})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or "")
        sys.stderr.write(exc.stdout or "")
        raise
