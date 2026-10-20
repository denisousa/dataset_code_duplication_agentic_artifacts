"""
Scan merged pull requests for code-clone / code-duplication discussions.

Workflow:
  1. Load the list of repositories and their merged PRs from the true
     configuration artifact PR results (7_true_configuration_artifact_prs_results).
  2. For each repository, fetch all merged pull requests in bulk via the GitHub
     REST list endpoint (100 per page), matching against the known PR numbers.
  3. For each PR, scan the body (description) locally against CLONE_TERMS.
  4. For PRs whose body matches, fetch general comments and review summaries,
     then scan those too.
  5. Record each match with its location (body / comment / review), the author,
     author_type (human / bot), and a text excerpt.

Outputs:
  - 10_clone_term_prs_results/clone_term_prs.json
  - 10_clone_term_prs_results/clone_term_prs_checkpoint.json

Notes:
  - Only PRs already present in the 7_ merged-PR list are considered.
  - Supports one or more GitHub tokens through GITHUB_TOKEN and/or
    GITHUB_TOKEN_1, GITHUB_TOKEN_2, ... in the environment (or .env file).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
PR_RESULTS_PATH = (
    ROOT_DIR
    / "7_true_configuration_artifact_prs_results"
    / "merged_prs_after_true_configuration_artifacts.json"
)

OUTPUT_DIR = ROOT_DIR / "10_clone_term_prs_results"
RESULTS_PATH = OUTPUT_DIR / "clone_term_prs.json"
CHECKPOINT_PATH = OUTPUT_DIR / "clone_term_prs_checkpoint.json"

GITHUB_API_BASE = "https://api.github.com"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_TOKEN_PREFIX = "GITHUB_TOKEN_"
REQUEST_TIMEOUT_SECONDS = 60
PER_PAGE = 100
SNIPPET_CONTEXT = 150
CHECKPOINT_SAVE_EVERY = 1

MAX_REPOS: int | None = int(os.getenv("MAX_REPOS", "0") or "0") or None

CLONE_TERMS: list[str] = [
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

_BOT_LOGIN_RE = re.compile(r"\[bot\]$|^(dependabot|renovate|github-actions|codecov|snyk|deepsource|codeclimate|mergify|allcontributors|semantic-release|release-drafter|greenkeeper|imgbot|whitesource|sonarqubebot|stale|netlify)(\[bot\])?$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    temp_path.replace(path)


# ---------------------------------------------------------------------------
# GitHub token pool
# ---------------------------------------------------------------------------

def base_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dataset-code-duplication-clone-term-pr-scan/1.0",
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
            earliest = min(float(e.get("available_at") or 0) for e in self.token_entries)
            wait = max(int(earliest - now), 0) + 1
            print(f"[RATE LIMIT] All tokens cooling down. Waiting {wait}s...")
            time.sleep(wait)

    def mark_rate_limited(self, entry: dict[str, Any], response: requests.Response) -> None:
        reset = response.headers.get("X-RateLimit-Reset", "")
        reset_at = int(reset) if reset else int(time.time()) + 60
        entry["available_at"] = float(reset_at + 5)

    def mark_temporarily_unavailable(self, entry: dict[str, Any], seconds: int) -> None:
        entry["available_at"] = float(time.time() + seconds)

    def has_available(self) -> bool:
        now = time.time()
        return any(float(e.get("available_at") or 0) <= now for e in self.token_entries)


def build_token_pool() -> GitHubTokenPool:
    entries: list[dict[str, Any]] = []

    direct = os.getenv(GITHUB_TOKEN_ENV, "").strip()
    if direct:
        entries.append({"name": GITHUB_TOKEN_ENV, "token": direct, "available_at": 0.0})

    numbered: list[tuple[int, str, str]] = []
    for name, value in os.environ.items():
        if not name.startswith(GITHUB_TOKEN_PREFIX):
            continue
        v = value.strip()
        if not v:
            continue
        suffix = name.removeprefix(GITHUB_TOKEN_PREFIX)
        if suffix.isdigit():
            numbered.append((int(suffix), name, v))
    for _, name, v in sorted(numbered):
        entries.append({"name": name, "token": v, "available_at": 0.0})

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
            msg = str(response.json().get("message") or "").lower()
        except ValueError:
            msg = ""
        if "secondary rate limit" not in msg:
            return False
        wait = max(retry_after, 120)
        if entry:
            pool.mark_temporarily_unavailable(entry, wait + 5)
        print(f"[RATE LIMIT] Secondary rate limit. Waiting {wait}s...")
        time.sleep(wait)
        return True

    if entry:
        pool.mark_rate_limited(entry, response)
        if pool.has_available():
            print(f"[RATE LIMIT] Token {entry['name']} exhausted. Switching...")
            return True
        wait = max(int(reset) - int(time.time()), 0) + 5
        print(f"[RATE LIMIT] All tokens exhausted. Waiting {wait}s...")
        time.sleep(wait)
        return True

    wait = max(int(reset) - int(time.time()), 0) + 5
    print(f"[RATE LIMIT] Waiting {wait}s...")
    time.sleep(wait)
    return True


def github_get(url: str, pool: GitHubTokenPool, *, params: dict | None = None) -> requests.Response:
    while True:
        headers, entry = pool.next_headers()
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if should_retry(response, pool, entry):
            continue
        return response


# ---------------------------------------------------------------------------
# Text scanning
# ---------------------------------------------------------------------------

def is_bot(login: str) -> bool:
    return bool(_BOT_LOGIN_RE.search(login))


def author_type(login: str) -> str:
    return "bot" if is_bot(login) else "human"


def extract_snippets(text: str, term: str) -> list[str]:
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    snippets: list[str] = []
    seen: set[str] = set()
    for m in pattern.finditer(text):
        start = max(0, m.start() - SNIPPET_CONTEXT)
        end = min(len(text), m.end() + SNIPPET_CONTEXT)
        snippet = "..." + text[start:end].replace("\n", " ").strip() + "..."
        if snippet not in seen:
            seen.add(snippet)
            snippets.append(snippet)
    return snippets


def scan_text(text: str, location: str, author: str) -> list[dict[str, Any]]:
    """Return one match entry per term found in text."""
    matches: list[dict[str, Any]] = []
    for term in CLONE_TERMS:
        snippets = extract_snippets(text, term)
        if snippets:
            matches.append(
                {
                    "location": location,
                    "author": author,
                    "author_type": author_type(author),
                    "term": term,
                    "excerpts": snippets[:3],
                }
            )
    return matches


# ---------------------------------------------------------------------------
# GitHub REST fetchers
# ---------------------------------------------------------------------------

def fetch_all_pages(url: str, pool: GitHubTokenPool, params: dict | None = None) -> list[dict]:
    """Paginate through a GitHub list endpoint, returning all items."""
    items: list[dict] = []
    page = 1
    base_params = dict(params or {})
    base_params["per_page"] = PER_PAGE

    while True:
        base_params["page"] = page
        response = github_get(url, pool, params=base_params)
        response.raise_for_status()
        batch = response.json()
        if not batch:
            break
        items.extend(batch)
        if len(batch) < PER_PAGE:
            break
        page += 1

    return items


def fetch_pr_body(repo: str, number: int, pool: GitHubTokenPool) -> str:
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{number}"
    response = github_get(url, pool)
    response.raise_for_status()
    return str(response.json().get("body") or "")


def fetch_issue_comments(repo: str, number: int, pool: GitHubTokenPool) -> list[dict]:
    url = f"{GITHUB_API_BASE}/repos/{repo}/issues/{number}/comments"
    return fetch_all_pages(url, pool)


def fetch_review_comments(repo: str, number: int, pool: GitHubTokenPool) -> list[dict]:
    """Fetch PR review summaries (not inline diff comments)."""
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{number}/reviews"
    return fetch_all_pages(url, pool)


# ---------------------------------------------------------------------------
# Per-PR scan
# ---------------------------------------------------------------------------

def scan_pr(
    repo: str,
    pr: dict[str, Any],
    pool: GitHubTokenPool,
) -> dict[str, Any] | None:
    """
    Scan a single PR. Returns a result dict if any CLONE_TERM was found,
    or None if the PR has no match.
    """
    number = pr["number"]
    author = pr.get("author_login", "")
    all_matches: list[dict[str, Any]] = []

    # 1. Fetch and scan body
    try:
        body = fetch_pr_body(repo, number, pool)
    except Exception as exc:
        print(f"    [WARN] Could not fetch body for #{number}: {exc}")
        body = ""

    if body:
        all_matches.extend(scan_text(body, "body", author))

    # 2. If body already matched, also scan comments/reviews
    #    If body didn't match, still scan comments (terms may be in discussions)
    try:
        comments = fetch_issue_comments(repo, number, pool)
    except Exception as exc:
        print(f"    [WARN] Could not fetch comments for #{number}: {exc}")
        comments = []

    for comment in comments:
        text = str(comment.get("body") or "")
        commenter = str((comment.get("user") or {}).get("login") or "")
        if text:
            all_matches.extend(scan_text(text, "comment", commenter))

    try:
        reviews = fetch_review_comments(repo, number, pool)
    except Exception as exc:
        print(f"    [WARN] Could not fetch reviews for #{number}: {exc}")
        reviews = []

    for review in reviews:
        text = str(review.get("body") or "")
        reviewer = str((review.get("user") or {}).get("login") or "")
        if text:
            all_matches.extend(scan_text(text, "review", reviewer))

    if not all_matches:
        return None

    matched_terms = sorted({m["term"] for m in all_matches})
    has_human_match = any(m["author_type"] == "human" for m in all_matches)

    return {
        "number": number,
        "title": pr.get("title", ""),
        "url": pr.get("url", ""),
        "merged_at": pr.get("merged_at", ""),
        "created_at": pr.get("created_at", ""),
        "author_login": author,
        "author_type": author_type(author),
        "matched_terms": matched_terms,
        "has_human_match": has_human_match,
        "matches": all_matches,
    }


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------

def process_repo(
    repo: str,
    known_prs: list[dict[str, Any]],
    pool: GitHubTokenPool,
) -> dict[str, Any]:
    known_by_number = {pr["number"]: pr for pr in known_prs}
    matching_prs: list[dict[str, Any]] = []
    scanned = 0

    print(f"  Scanning {len(known_prs)} PRs...")

    for pr in known_prs:
        scanned += 1
        number = pr["number"]
        result = scan_pr(repo, pr, pool)
        if result:
            matching_prs.append(result)
            terms_str = ", ".join(f'"{t}"' for t in result["matched_terms"])
            human_flag = " [human]" if result["has_human_match"] else " [bot only]"
            print(f"    #{number} MATCH{human_flag}: {terms_str}")

        if scanned % 50 == 0:
            print(f"    ... {scanned}/{len(known_prs)} scanned, {len(matching_prs)} matches so far")

    matching_prs.sort(key=lambda p: p.get("merged_at") or "")

    return {
        "status": "completed",
        "total_prs_scanned": scanned,
        "total_clone_term_prs": len(matching_prs),
        "total_human_match_prs": sum(1 for p in matching_prs if p["has_human_match"]),
        "prs": matching_prs,
    }


# ---------------------------------------------------------------------------
# Checkpoint / results
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "repos": {}}
    return load_json(path)


def build_summary(repos_state: dict[str, Any], repo_names: list[str]) -> dict[str, Any]:
    completed = [r for r in repo_names if repos_state.get(r, {}).get("status") == "completed"]
    failed = [r for r in repo_names if repos_state.get(r, {}).get("status") == "failed"]
    total_prs = sum(repos_state[r].get("total_clone_term_prs", 0) for r in completed)
    total_human = sum(repos_state[r].get("total_human_match_prs", 0) for r in completed)
    total_scanned = sum(repos_state[r].get("total_prs_scanned", 0) for r in completed)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_pr_results_path": str(PR_RESULTS_PATH),
        "clone_terms_searched": CLONE_TERMS,
        "total_repositories": len(repo_names),
        "repositories_completed": len(completed),
        "repositories_failed": len(failed),
        "total_prs_scanned": total_scanned,
        "total_clone_term_prs": total_prs,
        "total_human_match_prs": total_human,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pr_data = load_json(PR_RESULTS_PATH)
    repositories = pr_data.get("repositories", {})

    repo_names = sorted(repositories.keys())
    if MAX_REPOS is not None:
        repo_names = repo_names[:MAX_REPOS]

    pool = build_token_pool()
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    repos_state: dict[str, Any] = checkpoint.setdefault("repos", {})

    total_prs = sum(len(repositories[r].get("merged_prs", [])) for r in repo_names)
    print(f"Repositories to process: {len(repo_names)}")
    print(f"Total PRs to scan:       {total_prs}")
    print(f"Clone terms:             {len(CLONE_TERMS)}")

    processed_since_save = 0
    for index, repo in enumerate(repo_names, start=1):
        if repos_state.get(repo, {}).get("status") == "completed":
            print(f"[{index}/{len(repo_names)}] Reusing checkpoint for {repo}")
            continue

        known_prs = repositories[repo].get("merged_prs", [])
        print(f"[{index}/{len(repo_names)}] {repo} ({len(known_prs)} PRs)")

        try:
            result = process_repo(repo, known_prs, pool)
            repos_state[repo] = result
            print(
                f"  -> {result['total_clone_term_prs']} matching PR(s) "
                f"({result['total_human_match_prs']} with human mentions)"
            )
        except Exception as exc:
            repos_state[repo] = {"status": "failed", "error": str(exc), "prs": []}
            print(f"  -> FAILED: {exc}")

        processed_since_save += 1
        if processed_since_save >= CHECKPOINT_SAVE_EVERY:
            write_json(CHECKPOINT_PATH, checkpoint)
            processed_since_save = 0

    write_json(CHECKPOINT_PATH, checkpoint)

    results = {
        "summary": build_summary(repos_state, repo_names),
        "repositories": {r: repos_state[r] for r in repo_names if r in repos_state},
    }
    write_json(RESULTS_PATH, results)

    summary = results["summary"]
    print(f"\nRepositories completed:  {summary['repositories_completed']}")
    print(f"Repositories failed:     {summary['repositories_failed']}")
    print(f"Total PRs scanned:       {summary['total_prs_scanned']}")
    print(f"Total clone-term PRs:    {summary['total_clone_term_prs']}")
    print(f"  (human mentions only): {summary['total_human_match_prs']}")
    print(f"Results:   {RESULTS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()