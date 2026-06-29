"""
Collect AI-agent-attributed commits that happened after the creation date of
configuration artifacts reviewed as true evidence of code duplication.

Workflow:
  1. Load the reviewed JSON results and keep only configuration artifacts where
     is_real is True.
  2. Join those artifacts with the original clone-terms report to recover the
     file creation date (created_at). When created_at is unavailable, fall back
     to the artifact timestamp from the review payload.
  3. Load commits.csv and keep only commits whose ai_tool includes at least one
     of these agents: Claude, Gemini, Codex, Copilot, Cursor.
  4. For each repository with true configuration artifacts, collect commits on
     or after the earliest relevant artifact creation date.
  5. Attach the matching commits back to each artifact using that artifact's own
     effective creation date.

Outputs:
  - 9_true_configuration_artifact_agent_commits_results/agent_commits_after_true_configuration_artifacts.json
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
REVIEW_PATH = ROOT_DIR / "5_export_review_sources_results" / "gpt_5_mini_artifacts_and_commits_review.json"
CONFIG_REPORT_PATH = ROOT_DIR / "1_configurations_artifacts_results" / "clone_terms_report.json"
COMMITS_CSV_PATH = ROOT_DIR / "ai_config" / "commits.csv"

OUTPUT_DIR = ROOT_DIR / "9_true_configuration_artifact_agent_commits_results"
RESULTS_PATH = OUTPUT_DIR / "agent_commits_after_true_configuration_artifacts.json"

TARGET_AGENTS = {"Claude", "Gemini", "Codex", "Copilot", "Cursor"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def parse_iso8601(value: str) -> datetime | None:
    raw_value = value.strip()
    if not raw_value:
        return None

    normalized = raw_value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_utc_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_created_at_index(report_payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}

    for source in report_payload.get("sources", []):
        csv_name = str(source.get("csv") or "")
        artifact_type = csv_name.removesuffix(".csv") or "unknown"
        for match in source.get("matches", []):
            raw_url = str(match.get("raw_url") or "").strip()
            if not raw_url:
                continue
            index[raw_url] = {
                "created_at": str(match.get("created_at") or "").strip(),
                "timestamp": str(match.get("timestamp") or "").strip(),
                "first_commit_sha": str(match.get("first_commit_sha") or "").strip(),
                "last_commit_sha": str(match.get("last_commit_sha") or "").strip(),
                "artifact_type": artifact_type,
            }

    return index


def select_true_configuration_artifacts(
    review_payload: dict[str, Any],
    created_at_index: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []

    for result in review_payload.get("results", []):
        if result.get("source_kind") != "configuration_artifact":
            continue
        if result.get("is_real") is not True:
            continue

        raw_url = str(result.get("raw_url") or result.get("item_id") or "").strip()
        source_row = created_at_index.get(raw_url, {})

        created_at = str(source_row.get("created_at") or "").strip()
        fallback_timestamp = str(source_row.get("timestamp") or result.get("timestamp") or "").strip()
        effective_dt = parse_iso8601(created_at) or parse_iso8601(fallback_timestamp)
        if effective_dt is None:
            continue

        artifacts.append(
            {
                "repo_name": str(result.get("repo_name") or "").strip(),
                "artifact_type": str(result.get("artifact_type") or "").strip(),
                "source_group": str(result.get("source_group") or "").strip(),
                "item_id": str(result.get("item_id") or "").strip(),
                "url": str(result.get("url") or "").strip(),
                "raw_url": raw_url,
                "branch": str(result.get("branch") or "").strip(),
                "timestamp": str(result.get("timestamp") or "").strip(),
                "created_at": created_at,
                "effective_created_at": to_utc_z(effective_dt),
                "file_name": str(result.get("file_name") or "").strip(),
                "file_path": str(result.get("file_path") or "").strip(),
                "matched_term": str(result.get("matched_term") or "").strip(),
                "classification": str(result.get("classification") or "").strip(),
                "reason": str(result.get("reason") or "").strip(),
                "first_commit_sha": str(source_row.get("first_commit_sha") or "").strip(),
                "last_commit_sha": str(source_row.get("last_commit_sha") or "").strip(),
            }
        )

    artifacts.sort(key=lambda item: (item["repo_name"].lower(), item["effective_created_at"], item["file_path"].lower()))
    return artifacts


def parse_ai_tools(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(",") if part.strip()]


def keep_target_agent_commit(row: dict[str, str]) -> bool:
    tools = set(parse_ai_tools(str(row.get("ai_tool") or "")))
    return bool(tools & TARGET_AGENTS)


def commit_title(message: str) -> str:
    for line in message.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if line.strip():
            return line.strip()
    return ""


def load_agent_commits() -> list[dict[str, Any]]:
    csv.field_size_limit(10 * 1024 * 1024)
    commits: list[dict[str, Any]] = []

    with COMMITS_CSV_PATH.open("r", encoding="utf-8", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            if not keep_target_agent_commit(row):
                continue

            commit_timestamp = str(row.get("commit_timestamp") or "").strip()
            parsed_timestamp = parse_iso8601(commit_timestamp)
            if parsed_timestamp is None:
                continue

            ai_tool = str(row.get("ai_tool") or "").strip()
            commits.append(
                {
                    "repo_name": str(row.get("repo_name") or "").strip(),
                    "commit_sha": str(row.get("commit_sha") or "").strip(),
                    "url": str(row.get("github_link") or "").strip(),
                    "branch": str(row.get("branch") or "").strip(),
                    "commit_timestamp": to_utc_z(parsed_timestamp),
                    "ai_tool": ai_tool,
                    "agents": parse_ai_tools(ai_tool),
                    "commit_message_title": commit_title(str(row.get("commit_message") or "")),
                }
            )

    commits.sort(key=lambda item: (item["repo_name"].lower(), item["commit_timestamp"], item["commit_sha"]))
    return commits


def build_repository_results(
    artifacts_by_repo: dict[str, list[dict[str, Any]]],
    commits_by_repo: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    repository_results: dict[str, dict[str, Any]] = {}

    for repo_name, artifacts in artifacts_by_repo.items():
        repo_cutoff = min(artifact["effective_created_at"] for artifact in artifacts)
        matching_commits = [
            commit
            for commit in commits_by_repo.get(repo_name, [])
            if commit["commit_timestamp"] >= repo_cutoff
        ]

        repository_results[repo_name] = {
            "search_cutoff_timestamp": repo_cutoff,
            "matched_commit_count": len(matching_commits),
            "matched_commits": matching_commits,
        }

    return repository_results


def build_artifact_results(
    artifacts: list[dict[str, Any]],
    repository_results: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    artifact_results: list[dict[str, Any]] = []

    for artifact in artifacts:
        repo_state = repository_results.get(artifact["repo_name"], {})
        repo_commits = repo_state.get("matched_commits") or []
        cutoff = artifact["effective_created_at"]
        matching_commits = [
            commit
            for commit in repo_commits
            if commit["commit_timestamp"] >= cutoff
        ]

        artifact_results.append(
            {
                **artifact,
                "matched_commit_count_after_created_at": len(matching_commits),
                "matched_commit_shas_after_created_at": [commit["commit_sha"] for commit in matching_commits],
                "first_commit_after_created_at": matching_commits[0]["commit_timestamp"] if matching_commits else "",
                "last_commit_after_created_at": matching_commits[-1]["commit_timestamp"] if matching_commits else "",
            }
        )

    return artifact_results


def build_summary(
    artifacts: list[dict[str, Any]],
    agent_commits: list[dict[str, Any]],
    repository_results: dict[str, dict[str, Any]],
    artifact_results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_review_path": str(REVIEW_PATH),
        "source_configuration_report_path": str(CONFIG_REPORT_PATH),
        "source_commits_csv_path": str(COMMITS_CSV_PATH),
        "target_agents": sorted(TARGET_AGENTS),
        "total_true_configuration_artifacts": len(artifacts),
        "unique_repositories": len(repository_results),
        "total_agent_commits_in_csv": len(agent_commits),
        "total_repository_level_matched_commits": sum(
            repo_state["matched_commit_count"]
            for repo_state in repository_results.values()
        ),
        "total_artifact_commit_links": sum(
            artifact["matched_commit_count_after_created_at"]
            for artifact in artifact_results
        ),
    }


def main() -> None:
    review_payload = load_json(REVIEW_PATH)
    configuration_report = load_json(CONFIG_REPORT_PATH)
    created_at_index = build_created_at_index(configuration_report)
    artifacts = select_true_configuration_artifacts(review_payload, created_at_index)

    artifacts_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        artifacts_by_repo[artifact["repo_name"]].append(artifact)

    agent_commits = load_agent_commits()
    commits_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for commit in agent_commits:
        commits_by_repo[commit["repo_name"]].append(commit)

    repository_results = build_repository_results(artifacts_by_repo, commits_by_repo)
    artifact_results = build_artifact_results(artifacts, repository_results)

    payload = {
        "summary": build_summary(artifacts, agent_commits, repository_results, artifact_results),
        "repositories": repository_results,
        "artifacts": artifact_results,
    }
    write_json(RESULTS_PATH, payload)

    summary = payload["summary"]
    print(f"True configuration artifacts: {summary['total_true_configuration_artifacts']}")
    print(f"Unique repositories: {summary['unique_repositories']}")
    print(f"Agent-attributed commits in commits.csv: {summary['total_agent_commits_in_csv']}")
    print(f"Repository-level matched commits: {summary['total_repository_level_matched_commits']}")
    print(f"Artifact-level commit links: {summary['total_artifact_commit_links']}")
    print(f"Results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()