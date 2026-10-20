from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT_DIR = Path(__file__).resolve().parent
DIFFS_DIR = ROOT_DIR / "5_export_review_sources_results" / "commit_sources"
OUTPUT_DIR = ROOT_DIR / "8_commit_diff_language_analysis_results"
SUMMARY_PATH = OUTPUT_DIR / "commit_diff_language_summary.json"

DIFF_GLOB = "*.diff"
DIFF_HEADER_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+)$")

LANGUAGE_BY_EXTENSION = {
    ".py": "Python",
    ".pyi": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".mjs": "JavaScript",
    ".cjs": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".kts": "Kotlin",
    ".go": "Go",
    ".rs": "Rust",
    ".cs": "C#",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".hpp": "C++",
    ".hh": "C++",
    ".hxx": "C++",
    ".c": "C",
    ".h": "C/C++ Header",
    ".php": "PHP",
    ".rb": "Ruby",
    ".swift": "Swift",
    ".scala": "Scala",
    ".sh": "Shell",
    ".bash": "Shell",
    ".zsh": "Shell",
    ".ps1": "PowerShell",
    ".sql": "SQL",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".scss": "SCSS",
    ".sass": "Sass",
    ".less": "Less",
    ".vue": "Vue",
    ".svelte": "Svelte",
    ".json": "JSON",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".xml": "XML",
    ".toml": "TOML",
    ".md": "Markdown",
    ".proto": "Protocol Buffers",
    ".dart": "Dart",
    ".r": "R",
    ".jl": "Julia",
    ".lua": "Lua",
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def iter_diff_files(diffs_dir: Path) -> Iterable[Path]:
    yield from diffs_dir.rglob(DIFF_GLOB)


def extract_repo_name(diff_file: Path, diffs_dir: Path) -> str | None:
    try:
        relative_path = diff_file.relative_to(diffs_dir)
    except ValueError:
        return None

    if not relative_path.parts:
        return None

    return relative_path.parts[0].replace("§", "/")


def extract_changed_paths(text: str) -> list[str]:
    changed_paths: list[str] = []
    for line in text.splitlines():
        match = DIFF_HEADER_PATTERN.match(line)
        if match is None:
            continue
        normalized = match.group(2)
        if normalized != "/dev/null":
            changed_paths.append(normalized)
    return changed_paths


def infer_language(path_str: str) -> str:
    suffix = Path(path_str).suffix.lower()
    if suffix in LANGUAGE_BY_EXTENSION:
        return LANGUAGE_BY_EXTENSION[suffix]
    if Path(path_str).name in {"Dockerfile", "Containerfile"}:
        return "Dockerfile"
    if Path(path_str).name == "Makefile":
        return "Makefile"
    return "Unknown"


def main() -> None:
    diff_files = sorted(iter_diff_files(DIFFS_DIR))
    total_exported_files = len(diff_files)
    unique_repositories = sorted(
        {
            repo_name
            for diff_file in diff_files
            for repo_name in [extract_repo_name(diff_file, DIFFS_DIR)]
            if repo_name
        }
    )

    changed_paths: list[str] = []
    changed_files_per_diff: list[int] = []
    files_with_diff_markers = 0
    files_without_diff_markers = 0

    for diff_file in diff_files:
        text = diff_file.read_text(encoding="utf-8", errors="replace")
        paths = extract_changed_paths(text)
        if paths:
            files_with_diff_markers += 1
            changed_files_per_diff.append(len(paths))
            changed_paths.extend(paths)
        else:
            files_without_diff_markers += 1

    unique_changed_paths = sorted(set(changed_paths))
    language_counts = Counter(infer_language(path) for path in unique_changed_paths)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "diffs_dir": str(DIFFS_DIR),
        "total_exported_diff_files": total_exported_files,
        "unique_repositories": len(unique_repositories),
        "files_with_git_diff_markers": files_with_diff_markers,
        "files_without_git_diff_markers": files_without_diff_markers,
        "total_changed_file_entries": len(changed_paths),
        "unique_changed_files": len(unique_changed_paths),
        "average_changed_files_per_diff": (
            round(sum(changed_files_per_diff) / len(changed_files_per_diff), 2)
            if changed_files_per_diff
            else 0.0
        ),
        "top_languages": [
            {"language": language, "files": count}
            for language, count in language_counts.most_common()
        ],
        "note": (
            "No real git diff markers were found in the exported commit_diffs files. "
            "Language counts are based only on paths extracted from 'diff --git' lines when present."
            if files_with_diff_markers == 0
            else "Language counts are inferred from file extensions found in git diff paths."
        ),
    }

    write_json(SUMMARY_PATH, summary)

    print(f"Total exported diff files: {total_exported_files}")
    print(f"Unique repositories: {len(unique_repositories)}")
    print(f"Files with git diff markers: {files_with_diff_markers}")
    print(f"Total changed file entries: {len(changed_paths)}")
    print(f"Unique changed files: {len(unique_changed_paths)}")
    if changed_files_per_diff:
        average_files = sum(changed_files_per_diff) / len(changed_files_per_diff)
        print(f"Average changed files per diff: {average_files:.2f}")
    if language_counts:
        print("Top languages:")
        for language, count in language_counts.most_common(10):
            print(f"  - {language}: {count}")
    else:
        print("Top languages: none detected")
    print(f"Summary written to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
