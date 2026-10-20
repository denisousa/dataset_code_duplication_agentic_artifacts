"""
Collect merged pull requests for configuration artifacts that were reviewed as
true evidence of code duplication.

Workflow:
  1. Load the reviewed JSON results and keep only configuration artifacts where
     is_real is True.
  2. Join those artifacts with the original clone-terms report to recover the
     file creation date (created_at). When created_at is unavailable, fall back
     to the artifact timestamp from the review payload.
  3. Group artifacts by repository.
  4. For each repository, query the GitHub Search API for all merged pull
     requests merged on or after the earliest relevant artifact creation date.
  5. Save repository-level PR results once, then attach matching PR numbers to
     each true artifact based on its own creation date.

Outputs:
  - 7_true_configuration_artifact_prs_results/merged_prs_after_true_configuration_artifacts.json
  - 7_true_configuration_artifact_prs_results/merged_prs_after_true_configuration_artifacts_checkpoint.json

Notes:
  - This script uses the GitHub Search API, which only exposes the first 1,000
    search results for a given query. Repositories above that limit are flagged
    as truncated in the output.
  - For search results filtered with is:merged, the closed_at field is used as
    the merge timestamp.
    - Supports one or more GitHub tokens through GITHUB_TOKEN and/or
        GITHUB_TOKEN_1, GITHUB_TOKEN_2, ... in the environment (or .env file).
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
REVIEW_PATH = ROOT_DIR / "5_export_review_sources_results" / "gpt_5_mini_artifacts_and_commits_review.json"
CONFIG_REPORT_PATH = ROOT_DIR / "1_configurations_artifacts_results" / "clone_terms_report.json"

OUTPUT_DIR = ROOT_DIR / "7_true_configuration_artifact_prs_results"
RESULTS_PATH = OUTPUT_DIR / "merged_prs_after_true_configuration_artifacts.json"
CHECKPOINT_PATH = OUTPUT_DIR / "merged_prs_after_true_configuration_artifacts_checkpoint.json"

GITHUB_SEARCH_URL = "https://api.github.com/search/issues"
GITHUB_REPO_API_URL = "https://api.github.com/repos/{repo_name}"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_TOKEN_PREFIX = "GITHUB_TOKEN_"
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 0.0
PER_PAGE = 100
CHECKPOINT_SAVE_EVERY = 1

MAX_REPOS: int | None = int(os.getenv("MAX_REPOS", "0") or "0") or None
MAX_ARTIFACTS: int | None = int(os.getenv("MAX_ARTIFACTS", "0") or "0") or None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")
    temp_path.replace(path)


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


def base_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dataset-code-duplication-pr-collector/1.0",
    }


class GitHubTokenPool:
    def __init__(self, token_entries: list[dict[str, Any]]):
        self.token_entries = token_entries
        self.cursor = 0

    def is_authenticated(self) -> bool:
        return bool(self.token_entries)

    def next_headers(self) -> tuple[dict[str, str], dict[str, Any] | None]:
        headers = base_headers()
        token_entry = self._next_token_entry()
        if token_entry is not None:
            headers["Authorization"] = f"Bearer {token_entry['token']}"
        return headers, token_entry

    def _next_token_entry(self) -> dict[str, Any] | None:
        if not self.token_entries:
            return None

        while True:
            now = time.time()
            for _ in range(len(self.token_entries)):
                entry = self.token_entries[self.cursor % len(self.token_entries)]
                self.cursor += 1
                if float(entry.get("available_at") or 0) <= now:
                    return entry

            earliest = min(float(entry.get("available_at") or 0) for entry in self.token_entries)
            wait_seconds = max(int(earliest - now), 0) + 1
            print(f"[RATE LIMIT] All GitHub tokens are cooling down. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)

    def mark_rate_limited(self, token_entry: dict[str, Any], response: requests.Response) -> None:
        reset_value = response.headers.get("X-RateLimit-Reset", "")
        reset_at = int(reset_value) if reset_value else int(time.time()) + 60
        token_entry["available_at"] = float(reset_at + 5)

    def mark_temporarily_unavailable(self, token_entry: dict[str, Any], wait_seconds: int) -> None:
        token_entry["available_at"] = float(time.time() + wait_seconds)

    def has_available_token(self) -> bool:
        if not self.token_entries:
            return False
        now = time.time()
        return any(float(entry.get("available_at") or 0) <= now for entry in self.token_entries)


def build_token_pool() -> GitHubTokenPool:
    token_entries: list[dict[str, Any]] = []

    direct_token = os.getenv(GITHUB_TOKEN_ENV, "").strip()
    if direct_token:
        token_entries.append({"name": GITHUB_TOKEN_ENV, "token": direct_token, "available_at": 0.0})

    numbered_tokens: list[tuple[int, str, str]] = []
    for env_name, env_value in os.environ.items():
        if not env_name.startswith(GITHUB_TOKEN_PREFIX):
            continue
        token_value = env_value.strip()
        if not token_value:
            continue
        suffix = env_name.removeprefix(GITHUB_TOKEN_PREFIX)
        if suffix.isdigit():
            numbered_tokens.append((int(suffix), env_name, token_value))

    for _, env_name, token_value in sorted(numbered_tokens):
        token_entries.append({"name": env_name, "token": token_value, "available_at": 0.0})

    if token_entries:
        print(f"Using {len(token_entries)} GitHub token(s) for API requests.")
    else:
        print(
            "[WARNING] No GitHub tokens found in GITHUB_TOKEN or GITHUB_TOKEN_<n>. "
            "GitHub Search API calls may fail quickly because of rate limits."
        )

    return GitHubTokenPool(token_entries)


def should_retry_after_rate_limit(
    response: requests.Response,
    token_pool: GitHubTokenPool,
    token_entry: dict[str, Any] | None,
) -> bool:
    if response.status_code not in {403, 429}:
        return False

    retry_after = int(response.headers.get("Retry-After", "0") or "0")
    remaining = response.headers.get("X-RateLimit-Remaining", "")
    reset_value = response.headers.get("X-RateLimit-Reset", "")
    if remaining != "0" or not reset_value:
        try:
            payload = response.json()
        except ValueError:
            payload = {}

        message = str(payload.get("message") or "").lower()
        if "secondary rate limit" not in message:
            return False

        wait_seconds = max(retry_after, 120)
        if token_entry is not None:
            token_pool.mark_temporarily_unavailable(token_entry, wait_seconds + 5)
        print(f"[RATE LIMIT] Secondary rate limit hit. Waiting {wait_seconds}s before retrying...")
        time.sleep(wait_seconds)
        return True

    if token_entry is not None:
        token_pool.mark_rate_limited(token_entry, response)
        if token_pool.has_available_token():
            print(f"[RATE LIMIT] Token {token_entry['name']} exhausted. Switching token...")
            return True

        wait_seconds = max(int(reset_value) - int(time.time()), 0) + 5
        print(
            f"[RATE LIMIT] Token {token_entry['name']} exhausted and no other token is available. "
            f"Waiting {wait_seconds}s before retrying..."
        )
        time.sleep(wait_seconds)
        return True

    wait_seconds = max(int(reset_value) - int(time.time()), 0) + 5
    print(f"[RATE LIMIT] Waiting {wait_seconds}s before retrying...\n")
    time.sleep(wait_seconds)
    return True


def github_get(
    url: str,
    token_pool: GitHubTokenPool,
    *,
    params: dict[str, Any] | None = None,
) -> requests.Response:
    while True:
        headers, token_entry = token_pool.next_headers()
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        if should_retry_after_rate_limit(response, token_pool, token_entry):
            continue
        return response


def resolve_canonical_repo_name(repo_name: str, token_pool: GitHubTokenPool) -> str:
    response = github_get(GITHUB_REPO_API_URL.format(repo_name=repo_name), token_pool)
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("full_name") or repo_name).strip()


def is_invalid_repo_search_response(response: requests.Response) -> bool:
    if response.status_code != 422:
        return False

    try:
        payload = response.json()
    except ValueError:
        return False

    for error in payload.get("errors") or []:
        message = str(error.get("message") or "")
        if "cannot be searched" in message:
            return True

    return False


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
    if MAX_ARTIFACTS is not None:
        artifacts = artifacts[:MAX_ARTIFACTS]
    return artifacts


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "repos": {}}
    return load_json(path)


def build_search_query(repo_name: str, merged_since: str) -> str:
    return f"repo:{repo_name} is:pr is:merged merged:>={merged_since}"


def fetch_merged_prs_for_repo(repo_name: str, merged_since: str, token_pool: GitHubTokenPool) -> dict[str, Any]:
    collected: list[dict[str, Any]] = []
    page = 1
    total_count = 0
    truncated = False
    query_repo_name = repo_name

    while True:
        params = {
            "q": build_search_query(query_repo_name, merged_since),
            "sort": "updated",
            "order": "desc",
            "per_page": PER_PAGE,
            "page": page,
        }

        while True:
            response = github_get(GITHUB_SEARCH_URL, token_pool, params=params)
            if is_invalid_repo_search_response(response) and page == 1:
                canonical_repo_name = resolve_canonical_repo_name(repo_name, token_pool)
                if canonical_repo_name != query_repo_name:
                    print(
                        f"[REPO REDIRECT] {repo_name} is now {canonical_repo_name}. "
                        "Retrying search with canonical repository name..."
                    )
                    query_repo_name = canonical_repo_name
                    params["q"] = build_search_query(query_repo_name, merged_since)
                    continue

            if should_retry_after_rate_limit(response, token_pool, None):
                continue
            response.raise_for_status()
            break

        payload = response.json()
        if page == 1:
            total_count = int(payload.get("total_count") or 0)

        items = payload.get("items") or []
        for item in items:
            merged_at = str(item.get("closed_at") or "").strip()
            collected.append(
                {
                    "number": int(item.get("number") or 0),
                    "title": str(item.get("title") or "").strip(),
                    "url": str(item.get("html_url") or "").strip(),
                    "merged_at": merged_at,
                    "created_at": str(item.get("created_at") or "").strip(),
                    "updated_at": str(item.get("updated_at") or "").strip(),
                    "author_login": str((item.get("user") or {}).get("login") or "").strip(),
                }
            )

        fetched_so_far = page * PER_PAGE
        capped_total = min(total_count, 1000)
        if total_count > 1000 and fetched_so_far >= 1000:
            truncated = True
            break
        if len(items) < PER_PAGE or fetched_so_far >= capped_total:
            break

        page += 1
        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    collected.sort(key=lambda item: item.get("merged_at") or "")
    return {
        "status": "completed",
        "queried_repo_name": query_repo_name,
        "search_cutoff_timestamp": merged_since,
        "total_count": total_count,
        "truncated_by_search_limit": truncated,
        "merged_prs": collected,
    }


def build_artifact_results(
    artifacts: list[dict[str, Any]],
    checkpoint_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    artifact_results: list[dict[str, Any]] = []
    repos_state = checkpoint_payload.get("repos") or {}

    for artifact in artifacts:
        repo_state = repos_state.get(artifact["repo_name"], {})
        repo_prs = repo_state.get("merged_prs") or []
        cutoff = artifact["effective_created_at"]
        matching_prs = [pr for pr in repo_prs if str(pr.get("merged_at") or "") >= cutoff]

        artifact_results.append(
            {
                **artifact,
                "merged_pr_count_after_created_at": len(matching_prs),
                "merged_pr_numbers_after_created_at": [pr["number"] for pr in matching_prs],
                "first_merged_pr_after_created_at": matching_prs[0]["merged_at"] if matching_prs else "",
                "last_merged_pr_after_created_at": matching_prs[-1]["merged_at"] if matching_prs else "",
            }
        )

    return artifact_results


def build_summary(artifact_results: list[dict[str, Any]], checkpoint_payload: dict[str, Any]) -> dict[str, Any]:
    repos_state = checkpoint_payload.get("repos") or {}
    completed_repos = [name for name, state in repos_state.items() if state.get("status") == "completed"]
    failed_repos = [name for name, state in repos_state.items() if state.get("status") == "failed"]
    truncated_repos = [
        name
        for name, state in repos_state.items()
        if state.get("status") == "completed" and state.get("truncated_by_search_limit")
    ]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_review_path": str(REVIEW_PATH),
        "source_configuration_report_path": str(CONFIG_REPORT_PATH),
        "total_true_configuration_artifacts": len(artifact_results),
        "unique_repositories": len({artifact["repo_name"] for artifact in artifact_results}),
        "repositories_completed": len(completed_repos),
        "repositories_failed": len(failed_repos),
        "repositories_truncated_by_search_limit": truncated_repos,
        "total_repository_level_merged_prs": sum(
            len((state.get("merged_prs") or []))
            for state in repos_state.values()
            if state.get("status") == "completed"
        ),
        "total_artifact_pr_links": sum(artifact["merged_pr_count_after_created_at"] for artifact in artifact_results),
    }


def main() -> None:
    review_payload = load_json(REVIEW_PATH)
    configuration_report = load_json(CONFIG_REPORT_PATH)
    created_at_index = build_created_at_index(configuration_report)
    artifacts = select_true_configuration_artifacts(review_payload, created_at_index)

    artifacts_by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for artifact in artifacts:
        artifacts_by_repo[artifact["repo_name"]].append(artifact)

    repo_names = sorted(artifacts_by_repo)
    if MAX_REPOS is not None:
        repo_names = repo_names[:MAX_REPOS]
        artifacts = [artifact for artifact in artifacts if artifact["repo_name"] in set(repo_names)]

    token_pool = build_token_pool()
    checkpoint_payload = load_checkpoint(CHECKPOINT_PATH)
    repos_state = checkpoint_payload.setdefault("repos", {})

    print(f"True configuration artifacts: {len(artifacts)}")
    print(f"Repositories to process: {len(repo_names)}")

    processed_since_save = 0
    for index, repo_name in enumerate(repo_names, start=1):
        repo_artifacts = artifacts_by_repo[repo_name]
        repo_cutoff = min(artifact["effective_created_at"] for artifact in repo_artifacts)
        existing_state = repos_state.get(repo_name)

        if (
            existing_state
            and existing_state.get("status") == "completed"
            and existing_state.get("search_cutoff_timestamp") == repo_cutoff
        ):
            print(f"[{index}/{len(repo_names)}] Reusing checkpoint for {repo_name}")
            continue

        print(
            f"[{index}/{len(repo_names)}] Fetching merged PRs for {repo_name} "
            f"since {repo_cutoff}"
        )

        try:
            repos_state[repo_name] = fetch_merged_prs_for_repo(repo_name, repo_cutoff, token_pool)
            print(
                f"  -> {len(repos_state[repo_name].get('merged_prs') or [])} PRs collected"
                + (" (truncated at 1000 search results)" if repos_state[repo_name].get("truncated_by_search_limit") else "")
            )
        except Exception as exc:
            repos_state[repo_name] = {
                "status": "failed",
                "search_cutoff_timestamp": repo_cutoff,
                "error": str(exc),
                "merged_prs": [],
            }
            print(f"  -> FAILED: {exc}")

        processed_since_save += 1
        if processed_since_save >= CHECKPOINT_SAVE_EVERY:
            write_json(CHECKPOINT_PATH, checkpoint_payload)
            processed_since_save = 0

    write_json(CHECKPOINT_PATH, checkpoint_payload)

    artifact_results = build_artifact_results(artifacts, checkpoint_payload)
    results_payload = {
        "summary": build_summary(artifact_results, checkpoint_payload),
        "repositories": checkpoint_payload.get("repos") or {},
        "artifacts": artifact_results,
    }
    write_json(RESULTS_PATH, results_payload)

    summary = results_payload["summary"]
    print(f"Repositories completed: {summary['repositories_completed']}")
    print(f"Repositories failed: {summary['repositories_failed']}")
    print(f"Total repository-level merged PRs: {summary['total_repository_level_merged_prs']}")
    print(f"Total artifact PR links: {summary['total_artifact_pr_links']}")
    print(f"Results written to: {RESULTS_PATH}")
    print(f"Checkpoint written to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()