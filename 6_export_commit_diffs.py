from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = ROOT_DIR / "5_export_review_sources_results"
COMMITS_REVIEW_CSV_PATH = EXPORT_DIR / "commits_review.csv"
COMMITS_CSV_PATH = ROOT_DIR / "ai_config" / "commits.csv"
COMMIT_DIFFS_DIR = EXPORT_DIR / "commit_diffs"
COMMITS_WITH_DIFFS_CSV_PATH = EXPORT_DIR / "commits_review_with_diffs.csv"
DIFFS_MANIFEST_PATH = EXPORT_DIR / "commit_diffs_manifest.json"
REQUEST_DELAY_SECONDS = 0.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as file_handle:
            file_handle.write("")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def repo_dir_name(repo_name: str) -> str:
    return repo_name.replace("/", "§")


def load_commits_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    csv.field_size_limit(10 * 1024 * 1024)
    index: dict[tuple[str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            repo_name = str(row.get("repo_name") or "").strip()
            commit_sha = str(row.get("commit_sha") or "").strip()
            if repo_name and commit_sha:
                index[(repo_name, commit_sha)] = row
    return index


def build_commit_export_text(source_row: dict[str, str], review_row: dict[str, str]) -> str:
    commit_message = str(source_row.get("commit_message") or "")
    commit_message = commit_message.replace("\\r\\n", "\n").replace("\\n", "\n")

    lines = [
        f"repo_name: {review_row.get('repo_name', '')}",
        f"commit_sha: {review_row.get('commit_sha', '')}",
        f"github_link: {review_row.get('url', '')}",
        f"branch: {review_row.get('branch', '')}",
        f"timestamp: {review_row.get('timestamp', '')}",
        f"ai_tool: {review_row.get('ai_tool', '')}",
        f"matched_term: {review_row.get('matched_term', '')}",
        f"classification: {review_row.get('classification', '')}",
        "",
        "commit_message:",
        commit_message,
        "",
    ]
    return "\n".join(lines)


def export_diff_file(row: dict[str, str], commits_index: dict[tuple[str, str], dict[str, str]]) -> dict[str, Any]:
    repo_name = str(row.get("repo_name") or "")
    commit_sha = str(row.get("commit_sha") or "")
    export_path = COMMIT_DIFFS_DIR / repo_dir_name(repo_name) / f"{commit_sha}.txt"

    result = dict(row)
    result["diff_source_path"] = str(COMMITS_CSV_PATH)
    result["diff_source_column"] = "commit_message"
    result["diff_path"] = str(export_path)
    result["diff_exists"] = False
    result["diff_error"] = ""
    result["diff_bytes"] = 0

    try:
        source_row = commits_index.get((repo_name, commit_sha))
        if source_row is None:
            raise ValueError(f"Commit not found in commits.csv: {repo_name} / {commit_sha}")

        diff_text = build_commit_export_text(source_row, row)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(diff_text, encoding="utf-8")
        result["diff_exists"] = True
        result["diff_bytes"] = len(diff_text.encode("utf-8"))
    except Exception as exc:
        result["diff_error"] = str(exc)

    return result


def main() -> None:
    rows = load_rows(COMMITS_REVIEW_CSV_PATH)
    commits_index = load_commits_index(COMMITS_CSV_PATH)
    exported_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        repo_name = str(row.get("repo_name") or "")
        commit_sha = str(row.get("commit_sha") or "")
        print(f"[{index}/{len(rows)}] Exporting diff: {repo_name} / {commit_sha}")
        exported_rows.append(export_diff_file(row, commits_index))
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    write_csv(COMMITS_WITH_DIFFS_CSV_PATH, exported_rows)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(COMMITS_REVIEW_CSV_PATH),
        "output_csv": str(COMMITS_WITH_DIFFS_CSV_PATH),
        "diffs_dir": str(COMMIT_DIFFS_DIR),
        "total_rows": len(exported_rows),
        "exported_diffs": sum(1 for row in exported_rows if row.get("diff_exists")),
        "failed_diffs": sum(1 for row in exported_rows if not row.get("diff_exists")),
    }
    write_json(DIFFS_MANIFEST_PATH, manifest)

    print(f"Commit diffs exported: {manifest['exported_diffs']}")
    print(f"Commit diffs failed: {manifest['failed_diffs']}")
    print(f"Diff CSV: {COMMITS_WITH_DIFFS_CSV_PATH}")
    print(f"Diff directory: {COMMIT_DIFFS_DIR}")


if __name__ == "__main__":
    main()
