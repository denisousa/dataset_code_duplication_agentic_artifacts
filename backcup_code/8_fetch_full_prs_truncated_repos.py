"""
For repositories truncated by the GitHub Search API 1,000-result cap in
script 7, fetch ALL merged pull requests using the REST Pulls API.

  GET /repos/{owner}/{repo}/pulls?state=closed&sort=updated&direction=desc

This endpoint has no hard result cap.  Pagination stops early once every PR
on a page was last updated before the repo cutoff date, because
merged_at <= updated_at always holds — so those PRs cannot have
merged_at >= cutoff either.

Inputs:
  - 7_true_configuration_artifact_prs_results/merged_prs_after_true_configuration_artifacts.json

Outputs:
  - 8_full_merged_prs_truncated_repos_results/full_merged_prs_truncated_repos.json
  - 8_full_merged_prs_truncated_repos_results/full_merged_prs_truncated_repos_checkpoint.json

Notes:
  - Only repos flagged as truncated_by_search_limit in script 7 are processed.
  - The queried_repo_name recorded by script 7 is reused so renamed repos are
    queried under their current canonical name.
  - Supports one or more GitHub tokens through GITHUB_TOKEN and/or
    GITHUB_TOKEN_1, GITHUB_TOKEN_2, ... in the environment (or .env file).
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_PATH = (
    ROOT_DIR
    / "7_true_configuration_artifact_prs_results"
    / "merged_prs_after_true_configuration_artifacts.json"
)

OUTPUT_DIR = ROOT_DIR / "8_full_merged_prs_truncated_repos_results"
RESULTS_PATH = OUTPUT_DIR / "full_merged_prs_truncated_repos.json"
CHECKPOINT_PATH = OUTPUT_DIR / "full_merged_prs_truncated_repos_checkpoint.json"

GITHUB_PULLS_URL = "https://api.github.com/repos/{repo_name}/pulls"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_TOKEN_PREFIX = "GITHUB_TOKEN_"
REQUEST_TIMEOUT_SECONDS = 60
PER_PAGE = 100
CHECKPOINT_SAVE_EVERY = 1

MAX_REPOS: int | None = int(os.getenv("MAX_REPOS", "0") or "0") or None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "repos": {}}
    return load_json(path)


# ---------------------------------------------------------------------------
# GitHub auth / token pool
# ---------------------------------------------------------------------------

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

    def next_headers(self) -> tuple[dict[str, str], dict[str, Any] | None]:
        headers = base_headers()
        entry = self._next_entry()
        if entry is not None:
            headers["Authorization"] = f"Bearer {entry['token']}"
        return headers, entry

    def _next_entry(self) -> dict[str, Any] | None:
        if not self.token_entries:
            return None
        while True:
            now = time.time()
            for _ in range(len(self.token_entries)):
                entry = self.token_entries[self.cursor % len(self.token_entries)]
                self.cursor += 1
                if float(entry.get("available_at") or 0) <= now:
                    return entry
            earliest = min(float(e.get("available_at") or 0) for e in self.token_entries)
            wait = max(int(earliest - now), 0) + 1
            print(f"[RATE LIMIT] All tokens cooling down. Waiting {wait}s...")
            time.sleep(wait)

    def mark_rate_limited(self, entry: dict[str, Any], response: requests.Response) -> None:
        reset = response.headers.get("X-RateLimit-Reset", "")
        reset_at = int(reset) if reset else int(time.time()) + 60
        entry["available_at"] = float(reset_at + 5)

    def mark_temporarily_unavailable(self, entry: dict[str, Any], wait_seconds: int) -> None:
        entry["available_at"] = float(time.time() + wait_seconds)

    def has_available_token(self) -> bool:
        now = time.time()
        return any(float(e.get("available_at") or 0) <= now for e in self.token_entries)


def build_token_pool() -> GitHubTokenPool:
    entries: list[dict[str, Any]] = []

    direct = os.getenv(GITHUB_TOKEN_ENV, "").strip()
    if direct:
        entries.append({"name": GITHUB_TOKEN_ENV, "token": direct, "available_at": 0.0})

    numbered: list[tuple[int, str, str]] = []
    for env_name, env_value in os.environ.items():
        if not env_name.startswith(GITHUB_TOKEN_PREFIX):
            continue
        value = env_value.strip()
        if not value:
            continue
        suffix = env_name.removeprefix(GITHUB_TOKEN_PREFIX)
        if suffix.isdigit():
            numbered.append((int(suffix), env_name, value))
    for _, env_name, value in sorted(numbered):
        entries.append({"name": env_name, "token": value, "available_at": 0.0})

    if entries:
        print(f"Using {len(entries)} GitHub token(s).")
    else:
        print(
            "[WARNING] No GitHub tokens found in GITHUB_TOKEN or GITHUB_TOKEN_<n>. "
            "Rate limits may apply immediately."
        )

    return GitHubTokenPool(entries)


# ---------------------------------------------------------------------------
# HTTP + rate-limit handling
# ---------------------------------------------------------------------------

def should_retry(
    response: requests.Response,
    pool: GitHubTokenPool,
    entry: dict[str, Any] | None,
) -> bool:
    if response.status_code not in {403, 429}:
        return False

    retry_after = int(response.headers.get("Retry-After", "0") or "0")
    remaining = response.headers.get("X-RateLimit-Remaining", "")
    reset_value = response.headers.get("X-RateLimit-Reset", "")

    if remaining != "0" or not reset_value:
        try:
            msg = str((response.json().get("message") or "")).lower()
        except ValueError:
            msg = ""
        if "secondary rate limit" not in msg:
            return False
        wait = max(retry_after, 120)
        if entry is not None:
            pool.mark_temporarily_unavailable(entry, wait + 5)
        print(f"[RATE LIMIT] Secondary rate limit. Waiting {wait}s...")
        time.sleep(wait)
        return True

    if entry is not None:
        pool.mark_rate_limited(entry, response)
        if pool.has_available_token():
            print(f"[RATE LIMIT] Token {entry['name']} exhausted. Switching...")
            return True
        wait = max(int(reset_value) - int(time.time()), 0) + 5
        print(
            f"[RATE LIMIT] Token {entry['name']} exhausted and no other available. "
            f"Waiting {wait}s..."
        )
        time.sleep(wait)
        return True

    wait = max(int(reset_value) - int(time.time()), 0) + 5
    print(f"[RATE LIMIT] Waiting {wait}s...")
    time.sleep(wait)
    return True


def github_get(
    url: str,
    pool: GitHubTokenPool,
    *,
    params: dict[str, Any] | None = None,
) -> requests.Response:
    while True:
        headers, entry = pool.next_headers()
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if should_retry(response, pool, entry):
            continue
        return response


# ---------------------------------------------------------------------------
# PR fetching via REST Pulls API
# ---------------------------------------------------------------------------

def fetch_all_merged_prs(
    repo_name: str,
    cutoff: str,
    pool: GitHubTokenPool,
) -> dict[str, Any]:
    """
    Fetch every merged PR in repo_name with merged_at >= cutoff.

    Sorted by updated desc so we can stop as soon as all items on a page
    have updated_at < cutoff (since merged_at <= updated_at, those PRs
    cannot satisfy merged_at >= cutoff).
    """
    url = GITHUB_PULLS_URL.format(repo_name=repo_name)
    collected: list[dict[str, Any]] = []
    page = 1
    total_closed_fetched = 0

    while True:
        params: dict[str, Any] = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": PER_PAGE,
            "page": page,
        }
        response = github_get(url, pool, params=params)
        response.raise_for_status()

        items: list[dict[str, Any]] = response.json() or []
        total_closed_fetched += len(items)

        for item in items:
            merged_at = str(item.get("merged_at") or "").strip()
            if not merged_at:
                continue  # closed but not merged
            if merged_at < cutoff:
                continue  # merged before the cutoff window
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

        # Early exit: items are sorted by updated_at desc.  If the last item
        # on this page has updated_at < cutoff, every subsequent page will too,
        # and since merged_at <= updated_at those PRs cannot be in our window.
        if not items or str(items[-1].get("updated_at") or "") < cutoff:
            break

        if len(items) < PER_PAGE:
            break

        page += 1

    collected.sort(key=lambda pr: pr.get("merged_at") or "")
    return {
        "status": "completed",
        "search_cutoff_timestamp": cutoff,
        "total_closed_prs_fetched": total_closed_fetched,
        "merged_prs": collected,
    }


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def build_artifact_results(
    artifacts: list[dict[str, Any]],
    repos_state: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for artifact in artifacts:
        state = repos_state.get(artifact["repo_name"], {})
        repo_prs = state.get("merged_prs") or []
        cutoff = artifact["effective_created_at"]
        matching = [pr for pr in repo_prs if str(pr.get("merged_at") or "") >= cutoff]
        results.append(
            {
                **artifact,
                "merged_pr_count_after_created_at": len(matching),
                "merged_pr_numbers_after_created_at": [pr["number"] for pr in matching],
                "first_merged_pr_after_created_at": matching[0]["merged_at"] if matching else "",
                "last_merged_pr_after_created_at": matching[-1]["merged_at"] if matching else "",
            }
        )
    return results


def build_summary(
    artifact_results: list[dict[str, Any]],
    repos_state: dict[str, Any],
) -> dict[str, Any]:
    completed = [n for n, s in repos_state.items() if s.get("status") == "completed"]
    failed = [n for n, s in repos_state.items() if s.get("status") == "failed"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(SOURCE_PATH),
        "note": (
            "Full PR list via REST Pulls API for the repos that were truncated "
            "at 1,000 results by the GitHub Search API in script 7."
        ),
        "total_artifacts": len(artifact_results),
        "unique_repositories": len({a["repo_name"] for a in artifact_results}),
        "repositories_completed": len(completed),
        "repositories_failed": len(failed),
        "failed_repository_names": failed,
        "total_repository_level_merged_prs": sum(
            len(s.get("merged_prs") or [])
            for s in repos_state.values()
            if s.get("status") == "completed"
        ),
        "total_artifact_pr_links": sum(
            a["merged_pr_count_after_created_at"] for a in artifact_results
        ),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    source = load_json(SOURCE_PATH)

    repos_info: dict[str, Any] = source.get("repositories") or {}
    all_artifacts: list[dict[str, Any]] = source.get("artifacts") or []

    truncated_repo_names = sorted(
        name
        for name, state in repos_info.items()
        if state.get("status") == "completed" and state.get("truncated_by_search_limit")
    )

    if MAX_REPOS is not None:
        truncated_repo_names = truncated_repo_names[:MAX_REPOS]

    truncated_set = set(truncated_repo_names)
    artifacts = [a for a in all_artifacts if a.get("repo_name") in truncated_set]

    print(f"Truncated repositories from script 7: {len(truncated_repo_names)}")
    print(f"Artifacts in those repositories: {len(artifacts)}")

    pool = build_token_pool()
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    repos_state: dict[str, Any] = checkpoint.setdefault("repos", {})

    processed_since_save = 0
    for index, repo_name in enumerate(truncated_repo_names, start=1):
        repo_artifacts = [a for a in artifacts if a["repo_name"] == repo_name]
        cutoff = min(a["effective_created_at"] for a in repo_artifacts)

        existing = repos_state.get(repo_name)
        if (
            existing
            and existing.get("status") == "completed"
            and existing.get("search_cutoff_timestamp") == cutoff
        ):
            print(f"[{index}/{len(truncated_repo_names)}] Reusing checkpoint for {repo_name}")
            continue

        # Reuse the canonical name resolved by script 7 (handles renamed repos)
        api_repo_name = repos_info.get(repo_name, {}).get("queried_repo_name") or repo_name

        print(
            f"[{index}/{len(truncated_repo_names)}] Fetching all merged PRs for "
            f"{api_repo_name} since {cutoff}"
        )

        try:
            result = fetch_all_merged_prs(api_repo_name, cutoff, pool)
            repos_state[repo_name] = result
            print(
                f"  -> {len(result['merged_prs'])} merged PRs collected "
                f"(out of {result['total_closed_prs_fetched']} closed PRs fetched)"
            )
        except Exception as exc:
            repos_state[repo_name] = {
                "status": "failed",
                "search_cutoff_timestamp": cutoff,
                "error": str(exc),
                "merged_prs": [],
            }
            print(f"  -> FAILED: {exc}")

        processed_since_save += 1
        if processed_since_save >= CHECKPOINT_SAVE_EVERY:
            write_json(CHECKPOINT_PATH, checkpoint)
            processed_since_save = 0

    write_json(CHECKPOINT_PATH, checkpoint)

    artifact_results = build_artifact_results(artifacts, repos_state)
    payload = {
        "summary": build_summary(artifact_results, repos_state),
        "repositories": repos_state,
        "artifacts": artifact_results,
    }
    write_json(RESULTS_PATH, payload)

    summary = payload["summary"]
    print(f"Repositories completed:        {summary['repositories_completed']}")
    print(f"Repositories failed:           {summary['repositories_failed']}")
    if summary["failed_repository_names"]:
        for name in summary["failed_repository_names"]:
            print(f"  - {name}")
    print(f"Total merged PRs (repo level): {summary['total_repository_level_merged_prs']}")
    print(f"Total artifact PR links:       {summary['total_artifact_pr_links']}")
    print(f"Results written to:  {RESULTS_PATH}")
    print(f"Checkpoint written to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
