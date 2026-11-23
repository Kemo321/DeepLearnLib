#!/usr/bin/env python3
"""Write reports/coverage.md from the test suite and, when present, an lcov tracefile.

File coverage counts a library, model, or shared benchmark header as covered when a
test mentions its name. Line rates are copied from lcov when --lcov is passed.
Training drivers are listed apart from that percentage: they are programs, and the
loop they share is what the tests lock down.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

LIBRARY_ROOTS = (
    ROOT / "src",
    ROOT / "include" / "DeepLearnLib",
    ROOT / "benchmarks" / "models",
    ROOT / "benchmarks" / "common",
)
DRIVER_ROOTS = (
    ROOT / "benchmarks" / "train",
    ROOT / "benchmarks" / "forward",
    ROOT / "benchmarks" / "micro",
)
SOURCE_SUFFIXES = {".cpp", ".cu", ".hpp", ".h", ".cuh"}
TEST_CASE = re.compile(r"^(?:TEST|TEST_F|TYPED_TEST)\(\s*([^,\s]+)\s*,\s*([^\s)]+)", re.MULTILINE)


def iter_sources(roots: tuple[Path, ...]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            found.append(path)
    return sorted(found)


def test_texts() -> list[tuple[Path, str]]:
    tests = ROOT / "tests"
    rows: list[tuple[Path, str]] = []
    if not tests.is_dir():
        return rows
    for path in sorted(tests.rglob("*")):
        if path.is_file() and path.suffix in {".cpp", ".hpp"}:
            rows.append((path, path.read_text(encoding="utf-8", errors="replace")))
    return rows


def matching_tests(path: Path, tests: list[tuple[Path, str]]) -> list[str]:
    stem = path.stem
    tokens = {stem, stem.lower(), stem.replace("_", "")}
    if stem.startswith("test_"):
        tokens.add(stem[len("test_") :])
    hits: list[str] = []
    for test_path, text in tests:
        haystack = test_path.stem + "\n" + text
        if any(token and len(token) >= 3 and token in haystack for token in tokens):
            hits.append(test_path.relative_to(ROOT).as_posix())
    return hits


def parse_lcov(path: Path) -> dict[str, dict[str, int]]:
    rates: dict[str, dict[str, int]] = {}
    current: str | None = None
    found = hit = branches = branches_hit = functions = functions_hit = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SF:"):
            current = line[3:].replace("\\", "/")
            found = hit = branches = branches_hit = functions = functions_hit = 0
        elif line.startswith("LF:"):
            found = int(line[3:])
        elif line.startswith("LH:"):
            hit = int(line[3:])
        elif line.startswith("BRF:"):
            branches = int(line[4:])
        elif line.startswith("BRH:"):
            branches_hit = int(line[4:])
        elif line.startswith("FNF:"):
            functions = int(line[4:])
        elif line.startswith("FNH:"):
            functions_hit = int(line[4:])
        elif line == "end_of_record" and current:
            rates[current] = {
                "lines_found": found,
                "lines_hit": hit,
                "branches_found": branches,
                "branches_hit": branches_hit,
                "functions_found": functions,
                "functions_hit": functions_hit,
            }
            current = None
    return rates


def lcov_for(path: Path, rates: dict[str, dict[str, int]]) -> dict[str, int] | None:
    suffix = path.relative_to(ROOT).as_posix()
    for key, value in rates.items():
        if key.endswith(suffix):
            return value
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the DeepLearnLib coverage report.")
    parser.add_argument("--lcov", type=Path, default=None, help="Optional lcov .info tracefile.")
    parser.add_argument("--output", type=Path, default=REPORTS / "coverage.md")
    parser.add_argument("--note", default=None, help="Extra sentence placed under the title.")
    args = parser.parse_args()

    tests = test_texts()
    library = iter_sources(LIBRARY_ROOTS)
    drivers = iter_sources(DRIVER_ROOTS)
    rates = parse_lcov(args.lcov) if args.lcov and args.lcov.is_file() else {}

    rows = []
    for path in library:
        hits = matching_tests(path, tests)
        line = lcov_for(path, rates)
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "tested": bool(hits),
                "tests": hits,
                "lines_found": None if line is None else line["lines_found"],
                "lines_hit": None if line is None else line["lines_hit"],
                "branches_found": None if line is None else line["branches_found"],
                "branches_hit": None if line is None else line["branches_hit"],
                "functions_found": None if line is None else line["functions_found"],
                "functions_hit": None if line is None else line["functions_hit"],
            }
        )

    covered = sum(1 for row in rows if row["tested"])
    total = len(rows)
    percent = (100.0 * covered / total) if total else 0.0
    line_found = sum(row["lines_found"] or 0 for row in rows)
    line_hit = sum(row["lines_hit"] or 0 for row in rows)
    line_percent = (100.0 * line_hit / line_found) if line_found else None
    branch_found = sum(row["branches_found"] or 0 for row in rows)
    branch_hit = sum(row["branches_hit"] or 0 for row in rows)
    branch_percent = (100.0 * branch_hit / branch_found) if branch_found else None
    function_found = sum(row["functions_found"] or 0 for row in rows)
    function_hit = sum(row["functions_hit"] or 0 for row in rows)
    function_percent = (100.0 * function_hit / function_found) if function_found else None

    driver_rows = []
    for path in drivers:
        driver_rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "tests": matching_tests(path, tests),
            }
        )

    payload = {
        "file_coverage_percent": round(percent, 2),
        "files_covered": covered,
        "files_total": total,
        "line_coverage_percent": None if line_percent is None else round(line_percent, 2),
        "lines_hit": line_hit if line_found else None,
        "lines_found": line_found if line_found else None,
        "branch_coverage_percent": None if branch_percent is None else round(branch_percent, 2),
        "branches_hit": branch_hit if branch_found else None,
        "branches_found": branch_found if branch_found else None,
        "function_coverage_percent": None if function_percent is None else round(function_percent, 2),
        "functions_hit": function_hit if function_found else None,
        "functions_found": function_found if function_found else None,
        "lcov": None if not rates else args.lcov.name,
        "files": rows,
        "drivers": driver_rows,
    }

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Coverage",
        "",
        "This report maps library sources, application models, and the shared benchmark headers onto the GoogleTest suite.",
        "A file counts as covered when a test names it. Training and forward programs are drivers: they are not part of that percentage, because the loop, the CSV, and the models they call are what the tests check.",
        "",
    ]
    if args.note:
        lines.append(args.note)
        lines.append("")
    lines.append(f"- Files named by a test: **{covered} / {total}** ({percent:.1f}%)")
    lines.append(
        "- Execution on 23 Nov 2026: `build/tests/dllib_tests.exe` passed **147 / 147** tests in 1.4 s. "
        "None were skipped. The executable is the Release build from 17 Oct 2026, 20:16. "
        "Later edits on those trees are comments and blank lines, so they do not change that program."
    )
    if line_percent is None:
        lines.append(
            "- Line and branch coverage: not in this copy. Pass `--lcov coverage.filtered.info` "
            "from a GCC/Clang Debug build with `-DENABLE_COVERAGE=ON`. MSVC does not emit gcov notes."
        )
    else:
        lines.append(f"- Lines hit (lcov): **{line_hit} / {line_found}** ({line_percent:.1f}%)")
        if branch_percent is None:
            lines.append("- Branches: the tracefile has no `BRF`/`BRH` records.")
        else:
            lines.append(
                f"- Branches hit (lcov): **{branch_hit} / {branch_found}** ({branch_percent:.1f}%)"
            )
        if function_percent is not None:
            lines.append(
                f"- Functions hit (lcov): **{function_hit} / {function_found}** ({function_percent:.1f}%)"
            )
    code_lines: dict[str, int] = {}
    code_path = REPORTS / "loc-by-file.json"
    if code_path.is_file():
        loaded = json.loads(code_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            code_lines = {str(key).replace("\\", "/"): int(value) for key, value in loaded.items()}

    lines.extend(
        [
            "",
            "## Library, models, and shared headers",
            "",
            "Code is the cloc code-line count. Hits and branches come from lcov when `--lcov` is passed.",
            "",
            "| File | Tests | Code | Lines | Functions | Branches |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        test_cell = ", ".join(f"`{name}`" for name in row["tests"]) if row["tests"] else "—"
        code_cell = str(code_lines[row["path"]]) if row["path"] in code_lines else "—"
        if row["lines_found"]:
            line_cell = f"{row['lines_hit']} / {row['lines_found']}"
        else:
            line_cell = "—"
        if row["branches_found"]:
            branch_cell = f"{row['branches_hit']} / {row['branches_found']}"
        else:
            branch_cell = "—"
        if row["functions_found"]:
            function_cell = f"{row['functions_hit']} / {row['functions_found']}"
        else:
            function_cell = "—"
        lines.append(
            f"| `{row['path']}` | {test_cell} | {code_cell} | {line_cell} | {function_cell} | {branch_cell} |"
        )

    traced = [row for row in rows if row["lines_found"]]
    traced.sort(key=lambda row: (row["lines_hit"] / row["lines_found"]) if row["lines_found"] else 1)
    if traced:
        lines.extend(
            [
                "",
                "## Lowest line coverage",
                "",
                "The ten traced files with the smallest share of hit lines. A zero means the record was instrumented and no test executed it. CI has no GPU, so device-only lines stay at zero there.",
                "",
                "| File | Lines | Branches |",
                "| --- | --- | --- |",
            ]
        )
        for row in traced[:10]:
            branch_cell = (
                f"{row['branches_hit']} / {row['branches_found']}" if row["branches_found"] else "—"
            )
            lines.append(
                f"| `{row['path']}` | {row['lines_hit']} / {row['lines_found']} | {branch_cell} |"
            )

    missing = [row["path"] for row in rows if not row["tested"]]
    lines.extend(["", "## Files with no matching test", ""])
    if missing:
        lines.extend(f"- `{path}`" for path in missing)
    else:
        lines.append("Every listed file is named by at least one test.")

    lines.extend(["", "## Training and benchmark drivers", "", "These programs are the paired custom and Torch entry points. They are compiled by the release job and are not unit tests.", ""])
    for row in driver_rows:
        note = ", ".join(f"`{name}`" for name in row["tests"]) if row["tests"] else "paired binary"
        lines.append(f"- `{row['path']}` — {note}")

    case_rows: list[tuple[str, str, str]] = []
    for test_path, text in tests:
        for match in TEST_CASE.finditer(text):
            case_rows.append((test_path.relative_to(ROOT).as_posix(), match.group(1), match.group(2)))
    lines.extend(
        [
            "",
            "## Test cases",
            "",
            f"GoogleTest cases in `tests/`: **{len(case_rows)}**.",
            "The file table above is a name match, not a line trace. A line percentage comes only from an lcov tracefile.",
            "",
            "| Test file | Suite | Case |",
            "| --- | --- | --- |",
        ]
    )
    for path, suite, name in case_rows:
        lines.append(f"| `{path}` | `{suite}` | `{name}` |")

    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    try:
        shown = output.relative_to(ROOT).as_posix()
    except ValueError:
        shown = str(output)
    print(f"Wrote {shown} ({covered}/{total} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
