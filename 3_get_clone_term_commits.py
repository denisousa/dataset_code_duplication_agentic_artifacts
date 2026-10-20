"""
Collect commit messages that mention code-clone / code-duplication terms.

Workflow:
  1. Load every repository present in
     1_configurations_artifacts_results/clone_terms_report.json.
  2. For each repository, compute the earliest artifact created_at timestamp
     found in the report (falling back to timestamp when needed).
  3. Query the GitHub Search API for commit-message matches on or after that
     cutoff date, once per CLONE_TERM.
  4. Apply an exact timestamp filter locally so commits earlier than the
     created_at time on the same day are excluded.
  5. Deduplicate commits that match multiple CLONE_TERMS and write one record
     per unique commit.

Outputs:
    - clone_term_commits_results/clone_term_commits.json
    - clone_term_commits_results/clone_term_commits_checkpoint.json

Notes:
  - This script uses the GitHub Search API, which only exposes the first 1,000
    results for a given query. Term-level truncation is flagged in the output.
  - Commit search only covers the repository default branch.
  - Supports one or more GitHub tokens through GITHUB_TOKEN and/or
    GITHUB_TOKEN_1, GITHUB_TOKEN_2, ... in the environment (or .env file).
"""

from __future__ import annotations

import ast
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_REPORT_PATH = ROOT_DIR / "1_configurations_artifacts_results" / "clone_terms_report.json"
CONFIG_ARTIFACTS_SCRIPT_PATH = ROOT_DIR / "1_get_configurations_artifacts.py"

OUTPUT_DIR = ROOT_DIR / "clone_term_commits_results"
RESULTS_PATH = OUTPUT_DIR / "clone_term_commits.json"
CHECKPOINT_PATH = OUTPUT_DIR / "clone_term_commits_checkpoint.json"

CHECKPOINT_VERSION = 1
GITHUB_API_BASE = "https://api.github.com"
GITHUB_SEARCH_URL = f"{GITHUB_API_BASE}/search/commits"
GITHUB_REPO_API_URL = f"{GITHUB_API_BASE}/repos" + "/{repo_name}"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_TOKEN_PREFIX = "GITHUB_TOKEN_"
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 0.0
PER_PAGE = 100
SNIPPET_CONTEXT = 150
CHECKPOINT_SAVE_EVERY = 1

MAX_REPOS: int | None = int(os.getenv("MAX_REPOS", "0") or "0") or None

DEFAULT_CLONE_TERMS: list[str] = [
    "duplicate code",
    "duplicated code",
    "code duplication",
    "repeated code",
    "DRY principle",
    "reused code",
    "code reuse",
    "code sharing",
    "don't repeat yourself",
    "redundant code",
    "duplicate functionality",
    "duplicated functionality",
]


def load_clone_terms_from_script(path: Path) -> list[str]:
    try:
        source = path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError):
        return list(DEFAULT_CLONE_TERMS)

    for node in module.body:
        target_name = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    target_name = target.id
                    break
            value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value_node = node.value
        else:
            continue

        if target_name != "CLONE_TERMS" or value_node is None:
            continue

        try:
            parsed_terms = ast.literal_eval(value_node)
        except (ValueError, SyntaxError):
            return list(DEFAULT_CLONE_TERMS)

        if isinstance(parsed_terms, list) and all(isinstance(term, str) for term in parsed_terms):
            return parsed_terms
        return list(DEFAULT_CLONE_TERMS)

    return list(DEFAULT_CLONE_TERMS)


CLONE_TERMS: list[str] = load_clone_terms_from_script(CONFIG_ARTIFACTS_SCRIPT_PATH)


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


def iso_date(value: str) -> str:
    parsed = parse_iso8601(value)
    if parsed is None:
        return value[:10]
    return parsed.date().isoformat()


def base_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dataset-code-duplication-clone-term-commit-scan/1.0",
    }


class GitHubTokenPool:
    def __init__(self, token_entries: list[dict[str, Any]]):
        self.token_entries = token_entries
        self.cursor = 0

    def next_headers(self) -> tuple[dict[str, str], dict[str, Any] | None]:
        headers = base_headers()
        entry = self._next_entry()
        if entry:
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

            earliest = min(float(entry.get("available_at") or 0) for entry in self.token_entries)
            wait_seconds = max(int(earliest - now), 0) + 1
            print(f"[RATE LIMIT] All tokens cooling down. Waiting {wait_seconds}s...")
            time.sleep(wait_seconds)

    def mark_rate_limited(self, entry: dict[str, Any], response: requests.Response) -> None:
        reset = response.headers.get("X-RateLimit-Reset", "")
        reset_at = int(reset) if reset else int(time.time()) + 60
        entry["available_at"] = float(reset_at + 5)

    def mark_temporarily_unavailable(self, entry: dict[str, Any], seconds: int) -> None:
        entry["available_at"] = float(time.time() + seconds)

    def has_available(self) -> bool:
        now = time.time()
        return any(float(entry.get("available_at") or 0) <= now for entry in self.token_entries)


def build_token_pool() -> GitHubTokenPool:
    entries: list[dict[str, Any]] = []

    direct = os.getenv(GITHUB_TOKEN_ENV, "").strip()
    if direct:
        entries.append({"name": GITHUB_TOKEN_ENV, "token": direct, "available_at": 0.0})

    numbered: list[tuple[int, str, str]] = []
    for name, value in os.environ.items():
        if not name.startswith(GITHUB_TOKEN_PREFIX):
            continue
        token_value = value.strip()
        if not token_value:
            continue
        suffix = name.removeprefix(GITHUB_TOKEN_PREFIX)
        if suffix.isdigit():
            numbered.append((int(suffix), name, token_value))

    for _, name, token_value in sorted(numbered):
        entries.append({"name": name, "token": token_value, "available_at": 0.0})

    if entries:
        print(f"Using {len(entries)} GitHub token(s).")
    else:
        print("[WARNING] No GitHub tokens found. Rate limits will apply.")

    return GitHubTokenPool(entries)


def should_retry(
    response: requests.Response,
    pool: GitHubTokenPool,
    entry: dict[str, Any] | None,
) -> bool:
    if response.status_code not in {403, 429}:
        return False

    retry_after = int(response.headers.get("Retry-After", "0") or "0")
    remaining = response.headers.get("X-RateLimit-Remaining", "")
    reset = response.headers.get("X-RateLimit-Reset", "")

    if remaining != "0" or not reset:
        try:
            message = str(response.json().get("message") or "").lower()
        except ValueError:
            message = ""
        if "secondary rate limit" not in message:
            return False

        wait_seconds = max(retry_after, 120)
        if entry:
            pool.mark_temporarily_unavailable(entry, wait_seconds + 5)
        print(f"[RATE LIMIT] Secondary rate limit. Waiting {wait_seconds}s...")
        time.sleep(wait_seconds)
        return True

    if entry:
        pool.mark_rate_limited(entry, response)
        if pool.has_available():
            print(f"[RATE LIMIT] Token {entry['name']} exhausted. Switching...")
            return True
        wait_seconds = max(int(reset) - int(time.time()), 0) + 5
        print(f"[RATE LIMIT] All tokens exhausted. Waiting {wait_seconds}s...")
        time.sleep(wait_seconds)
        return True

    wait_seconds = max(int(reset) - int(time.time()), 0) + 5
    print(f"[RATE LIMIT] Waiting {wait_seconds}s...")
    time.sleep(wait_seconds)
    return True


def github_get(url: str, pool: GitHubTokenPool, *, params: dict[str, Any] | None = None) -> requests.Response:
    while True:
        headers, entry = pool.next_headers()
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if should_retry(response, pool, entry):
            continue
        return response


def build_repo_targets(report_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for source in report_payload.get("sources", []):
        source_csv = str(source.get("csv") or "").strip()
        artifact_type = source_csv.removesuffix(".csv") or "unknown"
        for match in source.get("matches", []):
            repo_name = str(match.get("repo_name") or "").strip()
            created_at = str(match.get("created_at") or "").strip()
            timestamp = str(match.get("timestamp") or "").strip()
            effective_dt = parse_iso8601(created_at) or parse_iso8601(timestamp)
            if not repo_name or effective_dt is None:
                continue

            targets[repo_name].append(
                {
                    "artifact_type": artifact_type,
                    "source_csv": source_csv,
                    "file_name": str(match.get("file_name") or "").strip(),
                    "file_path": str(match.get("file_path") or "").strip(),
                    "github_link": str(match.get("github_link") or "").strip(),
                    "raw_url": str(match.get("raw_url") or "").strip(),
                    "created_at": created_at,
                    "timestamp": timestamp,
                    "effective_created_at": to_utc_z(effective_dt),
                    "first_commit_sha": str(match.get("first_commit_sha") or "").strip(),
                    "last_commit_sha": str(match.get("last_commit_sha") or "").strip(),
                    "terms": match.get("terms") or [],
                }
            )

    repo_targets: dict[str, dict[str, Any]] = {}
    for repo_name, artifacts in targets.items():
        ordered_artifacts = sorted(
            artifacts,
            key=lambda item: (item["effective_created_at"], item["file_path"].lower(), item["file_name"].lower()),
        )
        repo_targets[repo_name] = {
            "repo_name": repo_name,
            "artifact_count": len(ordered_artifacts),
            "cutoff_created_at": ordered_artifacts[0]["effective_created_at"],
            "artifact_matches": ordered_artifacts,
        }

    return repo_targets


def extract_snippets(text: str, term: str) -> list[str]:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    snippets: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        start = max(0, match.start() - SNIPPET_CONTEXT)
        end = min(len(text), match.end() + SNIPPET_CONTEXT)
        snippet = "..." + text[start:end].replace("\n", " ").strip() + "..."
        if snippet not in seen:
            seen.add(snippet)
            snippets.append(snippet)
    return snippets


def scan_commit_message(message: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for term in CLONE_TERMS:
        snippets = extract_snippets(message, term)
        if snippets:
            matches.append(
                {
                    "term": term,
                    "count": len(re.findall(re.escape(term), message, flags=re.IGNORECASE)),
                    "snippets": snippets[:3],
                }
            )
    return matches


def resolve_canonical_repo_name(repo_name: str, pool: GitHubTokenPool) -> str:
    response = github_get(GITHUB_REPO_API_URL.format(repo_name=repo_name), pool)
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


def build_search_query(repo_name: str, term: str, committed_since_date: str) -> str:
    return f'repo:{repo_name} committer-date:>={committed_since_date} "{term}"'


def fetch_commits_for_term(
    repo_name: str,
    term: str,
    committed_since_date: str,
    pool: GitHubTokenPool,
) -> dict[str, Any]:
    collected: list[dict[str, Any]] = []
    page = 1
    total_count = 0
    truncated = False
    incomplete_results = False
    query_repo_name = repo_name

    while True:
        params = {
            "q": build_search_query(query_repo_name, term, committed_since_date),
            "sort": "committer-date",
            "order": "asc",
            "per_page": PER_PAGE,
            "page": page,
        }

        response = github_get(GITHUB_SEARCH_URL, pool, params=params)
        if is_invalid_repo_search_response(response) and page == 1:
            canonical_repo_name = resolve_canonical_repo_name(repo_name, pool)
            if canonical_repo_name != query_repo_name:
                print(
                    f"[REPO REDIRECT] {repo_name} is now {canonical_repo_name}. "
                    "Retrying search with canonical repository name..."
                )
                query_repo_name = canonical_repo_name
                continue

        response.raise_for_status()
        payload = response.json()
        if page == 1:
            total_count = int(payload.get("total_count") or 0)
            incomplete_results = bool(payload.get("incomplete_results"))

        items = payload.get("items") or []
        collected.extend(items)

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

    return {
        "queried_repo_name": query_repo_name,
        "term": term,
        "total_count": total_count,
        "truncated_by_search_limit": truncated,
        "incomplete_results": incomplete_results,
        "items": collected,
    }


def effective_commit_timestamp(item: dict[str, Any]) -> str:
    commit_payload = item.get("commit") or {}
    committer = commit_payload.get("committer") or {}
    author = commit_payload.get("author") or {}
    return str(committer.get("date") or author.get("date") or "").strip()


def build_commit_entry(item: dict[str, Any]) -> dict[str, Any] | None:
    commit_payload = item.get("commit") or {}
    message = str(commit_payload.get("message") or "")
    matches = scan_commit_message(message)
    if not matches:
        return None

    author = item.get("author") or {}
    committer = item.get("committer") or {}
    repository = item.get("repository") or {}
    commit_author = commit_payload.get("author") or {}
    commit_committer = commit_payload.get("committer") or {}

    return {
        "repo_name": str((repository.get("full_name") or "")).strip(),
        "commit_sha": str(item.get("sha") or "").strip(),
        "github_link": str(item.get("html_url") or "").strip(),
        "author_login": str(author.get("login") or "").strip(),
        "author_user_type": str(author.get("type") or "").strip(),
        "committer_login": str(committer.get("login") or "").strip(),
        "committer_user_type": str(committer.get("type") or "").strip(),
        "author_name": str(commit_author.get("name") or "").strip(),
        "author_email": str(commit_author.get("email") or "").strip(),
        "author_date": str(commit_author.get("date") or "").strip(),
        "committer_name": str(commit_committer.get("name") or "").strip(),
        "committer_email": str(commit_committer.get("email") or "").strip(),
        "committer_date": str(commit_committer.get("date") or "").strip(),
        "commit_message": message,
        "comment_count": int(commit_payload.get("comment_count") or 0),
        "matched_terms": sorted(match["term"] for match in matches),
        "total_occurrences": sum(int(match["count"]) for match in matches),
        "terms": matches,
    }


def merge_commit_entry(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged_terms: dict[str, dict[str, Any]] = {term_info["term"]: term_info for term_info in existing.get("terms", [])}
    for term_info in incoming.get("terms", []):
        merged_terms[term_info["term"]] = term_info

    existing["terms"] = sorted(merged_terms.values(), key=lambda item: item["term"])
    existing["matched_terms"] = sorted(merged_terms)
    existing["total_occurrences"] = sum(int(item["count"]) for item in existing["terms"])

    if not existing.get("repo_name"):
        existing["repo_name"] = incoming.get("repo_name", "")
    if not existing.get("github_link"):
        existing["github_link"] = incoming.get("github_link", "")
    return existing


def process_repo(repo: str, repo_target: dict[str, Any], pool: GitHubTokenPool) -> dict[str, Any]:
    cutoff_timestamp = repo_target["cutoff_created_at"]
    cutoff_date = iso_date(cutoff_timestamp)
    cutoff_dt = parse_iso8601(cutoff_timestamp)

    print(f"  Searching commit messages since {cutoff_timestamp}...")

    commits_by_sha: dict[str, dict[str, Any]] = {}
    term_searches: list[dict[str, Any]] = []
    queried_repo_name = repo

    for index, term in enumerate(CLONE_TERMS, start=1):
        search_result = fetch_commits_for_term(repo, term, cutoff_date, pool)
        queried_repo_name = search_result["queried_repo_name"]

        raw_items = search_result["items"]
        kept_for_term = 0
        for item in raw_items:
            effective_timestamp = effective_commit_timestamp(item)
            effective_dt = parse_iso8601(effective_timestamp)
            if cutoff_dt is not None and effective_dt is not None and effective_dt < cutoff_dt:
                continue

            entry = build_commit_entry(item)
            if entry is None:
                continue

            repo_name = entry.get("repo_name") or queried_repo_name
            if repo_name and repo_name != queried_repo_name:
                entry["repo_name"] = repo_name
            else:
                entry["repo_name"] = queried_repo_name

            sha = entry["commit_sha"]
            if sha in commits_by_sha:
                commits_by_sha[sha] = merge_commit_entry(commits_by_sha[sha], entry)
            else:
                commits_by_sha[sha] = entry
            kept_for_term += 1

        term_searches.append(
            {
                "term": term,
                "query_total_count": search_result["total_count"],
                "items_collected": len(raw_items),
                "items_kept_after_exact_cutoff": kept_for_term,
                "truncated_by_search_limit": search_result["truncated_by_search_limit"],
                "incomplete_results": search_result["incomplete_results"],
            }
        )

        if kept_for_term:
            print(f"    [{index}/{len(CLONE_TERMS)}] {term!r}: {kept_for_term} commit(s)")

    commits = sorted(
        commits_by_sha.values(),
        key=lambda item: (item.get("committer_date") or item.get("author_date") or "", item.get("commit_sha") or ""),
    )

    return {
        "status": "completed",
        "queried_repo_name": queried_repo_name,
        "search_cutoff_timestamp": cutoff_timestamp,
        "search_cutoff_date": cutoff_date,
        "artifact_count": repo_target["artifact_count"],
        "artifact_matches": repo_target["artifact_matches"],
        "total_clone_term_commits": len(commits),
        "total_search_queries": len(CLONE_TERMS),
        "queries_truncated_by_search_limit": [
            item["term"] for item in term_searches if item["truncated_by_search_limit"]
        ],
        "queries_with_incomplete_results": [
            item["term"] for item in term_searches if item["incomplete_results"]
        ],
        "term_searches": term_searches,
        "commits": commits,
    }


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": CHECKPOINT_VERSION, "repos": {}}

    payload = load_json(path)
    if payload.get("version") != CHECKPOINT_VERSION:
        return {"version": CHECKPOINT_VERSION, "repos": {}}
    return payload


def build_summary(repos_state: dict[str, Any], repo_names: list[str]) -> dict[str, Any]:
    completed = [repo for repo in repo_names if repos_state.get(repo, {}).get("status") == "completed"]
    failed = [repo for repo in repo_names if repos_state.get(repo, {}).get("status") == "failed"]
    total_commits = sum(repos_state[repo].get("total_clone_term_commits", 0) for repo in completed)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_configuration_report_path": str(CONFIG_REPORT_PATH),
        "clone_terms_searched": CLONE_TERMS,
        "total_repositories": len(repo_names),
        "repositories_completed": len(completed),
        "repositories_failed": len(failed),
        "repositories_with_truncated_queries": [
            repo for repo in completed if repos_state.get(repo, {}).get("queries_truncated_by_search_limit")
        ],
        "repositories_with_incomplete_search_results": [
            repo for repo in completed if repos_state.get(repo, {}).get("queries_with_incomplete_results")
        ],
        "total_clone_term_commits": total_commits,
    }


def main() -> None:
    report_payload = load_json(CONFIG_REPORT_PATH)
    repo_targets = build_repo_targets(report_payload)

    repo_names = sorted(repo_targets.keys())
    if MAX_REPOS is not None:
        repo_names = repo_names[:MAX_REPOS]

    token_pool = build_token_pool()
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    repos_state: dict[str, Any] = checkpoint.setdefault("repos", {})

    total_artifacts = sum(repo_targets[repo]["artifact_count"] for repo in repo_names)
    print(f"Repositories to process: {len(repo_names)}")
    print(f"Artifact matches loaded: {total_artifacts}")
    print(f"Clone terms:             {len(CLONE_TERMS)}")

    processed_since_save = 0
    for index, repo in enumerate(repo_names, start=1):
        repo_target = repo_targets[repo]
        existing_state = repos_state.get(repo, {})
        if (
            existing_state.get("status") == "completed"
            and existing_state.get("search_cutoff_timestamp") == repo_target["cutoff_created_at"]
        ):
            print(f"[{index}/{len(repo_names)}] Reusing checkpoint for {repo}")
            continue

        print(
            f"[{index}/{len(repo_names)}] {repo} "
            f"({repo_target['artifact_count']} artifact match(es), cutoff {repo_target['cutoff_created_at']})"
        )

        try:
            result = process_repo(repo, repo_target, token_pool)
            repos_state[repo] = result
            print(f"  -> {result['total_clone_term_commits']} matching commit(s)")
        except Exception as exc:
            repos_state[repo] = {
                "status": "failed",
                "search_cutoff_timestamp": repo_target["cutoff_created_at"],
                "error": str(exc),
                "commits": [],
            }
            print(f"  -> FAILED: {exc}")

        processed_since_save += 1
        if processed_since_save >= CHECKPOINT_SAVE_EVERY:
            write_json(CHECKPOINT_PATH, checkpoint)
            processed_since_save = 0

    write_json(CHECKPOINT_PATH, checkpoint)

    results = {
        "summary": build_summary(repos_state, repo_names),
        "repositories": {repo: repos_state[repo] for repo in repo_names if repo in repos_state},
    }
    write_json(RESULTS_PATH, results)

    summary = results["summary"]
    print(f"\nRepositories completed:  {summary['repositories_completed']}")
    print(f"Repositories failed:     {summary['repositories_failed']}")
    print(f"Total matching commits:  {summary['total_clone_term_commits']}")
    print(f"Results:   {RESULTS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()