from __future__ import annotations

import csv
import json
import logging
import os
import requests
import shutil
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
REVIEW_PATH = ROOT_DIR / "4_artifacts_and_commits_review_results" / "gpt_5_mini_artifacts_and_commits_review.json"
COMMITS_CSV_PATH = ROOT_DIR / "ai_config" / "commits.csv"
LOCAL_REPOS_DIR = ROOT_DIR / "ai_config" / "repos_data" / "repos"

OUTPUT_DIR = ROOT_DIR / "5_export_review_sources_results"
CONFIGURATION_CSV_PATH = OUTPUT_DIR / "configuration_artifacts_review.csv"
COMMITS_EXPORT_CSV_PATH = OUTPUT_DIR / "commits_review_with_diffs.csv"
REVIEW_EXPORT_PATH = OUTPUT_DIR / REVIEW_PATH.name
CONFIGURATION_SOURCES_DIR = OUTPUT_DIR / "configuration_artifacts"
COMMIT_SOURCES_DIR = OUTPUT_DIR / "commit_sources"
REQUEST_DELAY_SECONDS = 0.0
REQUEST_TIMEOUT_SECONDS = 60
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"

LOGGER = logging.getLogger(__name__)

CONFIGURATION_FIELDNAMES = [
    "source_kind",
    "artifact_type",
    "repo_name",
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
    "source_path",
]

COMMIT_FIELDNAMES = [
    "source_kind",
    "artifact_type",
    "repo_name",
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
    "comment_path",
    "diff_path",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def repo_dir_name(repo_name: str) -> str:
    return repo_name.replace("/", "§")


def relative_from_marker(path_str: str, marker: str) -> str:
    path = Path(path_str)
    parts = path.parts
    if marker in parts:
        marker_index = parts.index(marker)
        return str(Path(*parts[marker_index:]))
    return path_str


def select_fields(record: dict[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    return {field: record.get(field) for field in fieldnames}


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


def build_github_headers(accept: str) -> dict[str, str]:
    headers = {
        "Accept": accept,
        "User-Agent": "dataset-code-duplication-export/1.0",
    }
    token = os.getenv(GITHUB_TOKEN_ENV, "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def build_diff_url(commit_url: str) -> str:
    commit_url = commit_url.strip()
    if commit_url.endswith(".diff"):
        return commit_url
    return f"{commit_url}.diff"


def build_commit_api_url(repo_name: str, commit_sha: str) -> str:
    return f"https://api.github.com/repos/{repo_name}/commits/{commit_sha}"


def fetch_text(url: str, accept: str) -> str:
    response = requests.get(url, headers=build_github_headers(accept), timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def fetch_commit_diff_text(review_record: dict[str, Any]) -> str:
    repo_name = str(review_record.get("repo_name") or "").strip()
    commit_sha = str(review_record.get("commit_sha") or "").strip()
    commit_url = str(review_record.get("url") or "").strip()

    errors: list[str] = []

    if commit_url:
        diff_url = build_diff_url(commit_url)
        try:
            diff_text = fetch_text(diff_url, "text/plain")
            if "diff --git " in diff_text:
                return diff_text
            errors.append(f"No git diff markers in {diff_url}")
        except Exception as exc:
            errors.append(f"{diff_url}: {exc}")

    if repo_name and commit_sha:
        api_url = build_commit_api_url(repo_name, commit_sha)
        try:
            diff_text = fetch_text(api_url, "application/vnd.github.v3.diff")
            if "diff --git " in diff_text:
                return diff_text
            errors.append(f"No git diff markers in {api_url}")
        except Exception as exc:
            errors.append(f"{api_url}: {exc}")

    raise RuntimeError(" ; ".join(errors) if errors else "Unable to fetch commit diff.")


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


def extract_commit_message_text(review_record: dict[str, Any], source_row: dict[str, str] | None) -> str:
    commit_message = ""
    if source_row is not None:
        commit_message = str(source_row.get("commit_message") or "")
    if not commit_message:
        commit_message = str(review_record.get("commit_message") or "")
    return commit_message.replace("\\r\\n", "\n").replace("\\n", "\n")


def build_commit_message_path(review_record: dict[str, Any]) -> Path:
    repo_name = str(review_record.get("repo_name") or "")
    commit_sha = str(review_record.get("commit_sha") or "")
    return COMMIT_SOURCES_DIR / repo_dir_name(repo_name) / f"{commit_sha}.comment.txt"


def build_commit_diff_path(review_record: dict[str, Any]) -> Path:
    repo_name = str(review_record.get("repo_name") or "")
    commit_sha = str(review_record.get("commit_sha") or "")
    return COMMIT_SOURCES_DIR / repo_dir_name(repo_name) / f"{commit_sha}.diff"


def remove_legacy_commit_metadata_files() -> int:
    if not COMMIT_SOURCES_DIR.exists():
        return 0

    removed_count = 0
    for metadata_path in COMMIT_SOURCES_DIR.rglob("*.metadata.json"):
        metadata_path.unlink(missing_ok=True)
        removed_count += 1
    return removed_count


def remove_legacy_export_files() -> int:
    removed_count = 0
    for legacy_path in (
        OUTPUT_DIR / "export_manifest.json",
        OUTPUT_DIR / "export_checkpoint.json",
    ):
        if legacy_path.exists():
            legacy_path.unlink()
            removed_count += 1
    return removed_count


def export_commit_message(review_record: dict[str, Any], source_row: dict[str, str] | None) -> str:
    export_path = build_commit_message_path(review_record)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(extract_commit_message_text(review_record, source_row), encoding="utf-8")
    return str(export_path)


def export_commit_diff(review_record: dict[str, Any]) -> tuple[str, bool]:
    export_path = build_commit_diff_path(review_record)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    diff_text = fetch_commit_diff_text(review_record)
    export_path.write_text(diff_text, encoding="utf-8")
    return str(export_path), True


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_review_source() -> None:
    REVIEW_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REVIEW_PATH, REVIEW_EXPORT_PATH)


def main() -> None:
    configure_logging()
    payload = load_json(REVIEW_PATH)
    results = list(payload.get("results", []))
    commits_index = load_commits_index(COMMITS_CSV_PATH)
    removed_metadata_files = remove_legacy_commit_metadata_files()
    removed_export_files = remove_legacy_export_files()
    configuration_rows: list[dict[str, Any]] = []
    commit_rows: list[dict[str, Any]] = []
    exported_commit_diffs = 0
    failed_commit_diffs = 0

    LOGGER.info("Starting export for %s review results.", len(results))
    if removed_metadata_files:
        LOGGER.info("Removed %s legacy commit metadata files.", removed_metadata_files)
    if removed_export_files:
        LOGGER.info("Removed %s legacy export JSON files.", removed_export_files)

    for index, result in enumerate(results, start=1):
        source_kind = str(result.get("source_kind") or "")
        repo_name = str(result.get("repo_name") or "")

        if source_kind == "commit":
            item_label = str(result.get("commit_sha") or "")
        else:
            item_label = str(result.get("file_path") or result.get("url") or "")

        LOGGER.info("[%s/%s] Exporting %s: %s / %s", index, len(results), source_kind, repo_name, item_label)

        if source_kind == "configuration_artifact":
            original_source_path, export_source_path, source_exists = copy_configuration_source(result)
            configuration_row = {
                **result,
                "source_original_path": original_source_path,
                "source_path": relative_from_marker(export_source_path, "configuration_artifacts"),
                "source_exists": source_exists,
            }
            configuration_rows.append(select_fields(configuration_row, CONFIGURATION_FIELDNAMES))

        elif source_kind == "commit":
            commit_sha = str(result.get("commit_sha") or "")
            source_row = commits_index.get((repo_name, commit_sha))
            export_message_path = export_commit_message(result, source_row)
            diff_path = ""
            try:
                export_diff_path, _ = export_commit_diff(result)
                diff_path = relative_from_marker(export_diff_path, "commit_sources")
                exported_commit_diffs += 1
            except Exception as exc:
                failed_commit_diffs += 1
                LOGGER.warning("Diff export failed for %s / %s: %s", repo_name, commit_sha, exc)
            commit_row = {
                **result,
                "comment_path": relative_from_marker(export_message_path, "commit_sources"),
                "diff_path": diff_path,
            }
            commit_rows.append(select_fields(commit_row, COMMIT_FIELDNAMES))

        else:
            LOGGER.warning("Skipping unsupported source_kind=%r for repo=%s", source_kind, repo_name)

        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    write_csv(CONFIGURATION_CSV_PATH, configuration_rows, CONFIGURATION_FIELDNAMES)
    write_csv(COMMITS_EXPORT_CSV_PATH, commit_rows, COMMIT_FIELDNAMES)
    copy_review_source()

    LOGGER.info("Configuration artifact rows exported: %s", len(configuration_rows))
    LOGGER.info("Commit rows exported: %s", len(commit_rows))
    LOGGER.info("Configuration CSV: %s", CONFIGURATION_CSV_PATH)
    LOGGER.info("Commit CSV: %s", COMMITS_EXPORT_CSV_PATH)
    LOGGER.info("Review JSON copy: %s", REVIEW_EXPORT_PATH)
    LOGGER.info("Configuration artifacts: %s", CONFIGURATION_SOURCES_DIR)
    LOGGER.info("Commit sources: %s", COMMIT_SOURCES_DIR)


if __name__ == "__main__":
    main()
