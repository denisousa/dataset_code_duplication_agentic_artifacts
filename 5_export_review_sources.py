from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
REVIEW_PATH = ROOT_DIR / "4_artifacts_and_commits_review_results" / "gpt_5_mini_artifacts_and_commits_review.json"
COMMITS_CSV_PATH = ROOT_DIR / "ai_config" / "commits.csv"
LOCAL_REPOS_DIR = ROOT_DIR / "ai_config" / "repos_data" / "repos"

OUTPUT_DIR = ROOT_DIR / "5_export_review_sources_results"
CONFIGURATION_CSV_PATH = OUTPUT_DIR / "configuration_artifacts_review.csv"
COMMITS_EXPORT_CSV_PATH = OUTPUT_DIR / "commits_review.csv"
CONFIGURATION_SOURCES_DIR = OUTPUT_DIR / "configuration_artifact_sources"
COMMIT_SOURCES_DIR = OUTPUT_DIR / "commit_sources"
MANIFEST_PATH = OUTPUT_DIR / "export_manifest.json"

CONFIGURATION_FIELDNAMES = [
    "source_kind",
    "artifact_type",
    "repo_name",
    "item_id",
    "url",
    "raw_url",
    "branch",
    "timestamp",
    "file_name",
    "file_path",
    "matched_term",
    "is_real",
    "classification",
    "matched_term_response",
    "reason",
    "model_response_text",
    "content_length",
    "source_original_path",
    "source_path",
    "source_exists",
]

COMMIT_FIELDNAMES = [
    "source_kind",
    "artifact_type",
    "repo_name",
    "item_id",
    "original_item_id",
    "url",
    "branch",
    "timestamp",
    "commit_sha",
    "ai_tool",
    "matched_term",
    "is_real",
    "classification",
    "matched_term_response",
    "reason",
    "model_response_text",
    "content_length",
    "source_original_path",
    "source_path",
    "source_column",
    "source_found_in_commits_csv",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def repo_dir_name(repo_name: str) -> str:
    return repo_name.replace("/", "§")


def load_commits_index(commits_csv_path: Path) -> dict[tuple[str, str], dict[str, str]]:
    csv.field_size_limit(10 * 1024 * 1024)
    index: dict[tuple[str, str], dict[str, str]] = {}
    with commits_csv_path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            repo_name = str(row.get("repo_name") or "").strip()
            commit_sha = str(row.get("commit_sha") or "").strip()
            if repo_name and commit_sha:
                index[(repo_name, commit_sha)] = row
    return index


def copy_configuration_source(record: dict[str, Any]) -> tuple[str, str, bool]:
    repo_name = str(record.get("repo_name") or "")
    file_path = str(record.get("file_path") or "")
    source_path = LOCAL_REPOS_DIR / repo_dir_name(repo_name) / file_path
    export_path = CONFIGURATION_SOURCES_DIR / repo_dir_name(repo_name) / file_path

    if source_path.exists():
        export_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, export_path)
        return str(source_path), str(export_path), True

    return str(source_path), "", False


def build_commit_source_text(review_record: dict[str, Any], source_row: dict[str, str] | None) -> str:
    commit_message = ""
    if source_row is not None:
        commit_message = str(source_row.get("commit_message") or "")
    if not commit_message:
        commit_message = str(review_record.get("commit_message") or "")
    commit_message = commit_message.replace("\\r\\n", "\n").replace("\\n", "\n")

    lines = [
        f"repo_name: {review_record.get('repo_name', '')}",
        f"commit_sha: {review_record.get('commit_sha', '')}",
        f"github_link: {review_record.get('url', '')}",
        f"branch: {review_record.get('branch', '')}",
        f"timestamp: {review_record.get('timestamp', '')}",
        f"ai_tool: {review_record.get('ai_tool', '')}",
        f"matched_term: {review_record.get('matched_term', '')}",
        f"classification: {review_record.get('classification', '')}",
        "",
        "commit_message:",
        commit_message,
        "",
    ]
    return "\n".join(lines)


def export_commit_source(review_record: dict[str, Any], source_row: dict[str, str] | None) -> tuple[str, str]:
    repo_name = str(review_record.get("repo_name") or "")
    commit_sha = str(review_record.get("commit_sha") or "")
    export_path = COMMIT_SOURCES_DIR / repo_dir_name(repo_name) / f"{commit_sha}.txt"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(build_commit_source_text(review_record, source_row), encoding="utf-8")
    return str(COMMITS_CSV_PATH), str(export_path)


def select_fields(record: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    return {field: record.get(field) for field in fieldnames}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    payload = load_json(REVIEW_PATH)
    results = list(payload.get("results", []))
    commits_index = load_commits_index(COMMITS_CSV_PATH)

    configuration_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []

    for result in results:
        source_kind = str(result.get("source_kind") or "")

        if source_kind == "configuration_artifact":
            original_source_path, export_source_path, source_exists = copy_configuration_source(result)
            configuration_row = {
                **result,
                "source_original_path": original_source_path,
                "source_path": export_source_path,
                "source_exists": source_exists,
            }
            configuration_rows.append(select_fields(configuration_row, CONFIGURATION_FIELDNAMES))
            continue

        if source_kind == "commit":
            repo_name = str(result.get("repo_name") or "")
            commit_sha = str(result.get("commit_sha") or "")
            source_row = commits_index.get((repo_name, commit_sha))
            source_original_path, export_source_path = export_commit_source(result, source_row)
            commit_row = {
                **result,
                "source_original_path": source_original_path,
                "source_path": export_source_path,
                "source_column": "commit_message",
                "source_found_in_commits_csv": source_row is not None,
            }
            commit_rows.append(select_fields(commit_row, COMMIT_FIELDNAMES))

    write_csv(CONFIGURATION_CSV_PATH, configuration_rows, CONFIGURATION_FIELDNAMES)
    write_csv(COMMITS_EXPORT_CSV_PATH, commit_rows, COMMIT_FIELDNAMES)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_path": str(REVIEW_PATH),
        "configuration_csv": str(CONFIGURATION_CSV_PATH),
        "commits_csv": str(COMMITS_EXPORT_CSV_PATH),
        "configuration_sources_dir": str(CONFIGURATION_SOURCES_DIR),
        "commit_sources_dir": str(COMMIT_SOURCES_DIR),
        "configuration_rows": len(configuration_rows),
        "commit_rows": len(commit_rows),
    }
    write_json(MANIFEST_PATH, manifest)

    print(f"Configuration artifact rows exported: {len(configuration_rows)}")
    print(f"Commit rows exported: {len(commit_rows)}")
    print(f"Configuration CSV: {CONFIGURATION_CSV_PATH}")
    print(f"Commit CSV: {COMMITS_EXPORT_CSV_PATH}")
    print(f"Configuration sources: {CONFIGURATION_SOURCES_DIR}")
    print(f"Commit sources: {COMMIT_SOURCES_DIR}")


if __name__ == "__main__":
    main()
