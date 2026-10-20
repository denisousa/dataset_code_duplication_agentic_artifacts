"""
Scan merged pull requests for code-clone / code-duplication discussions.

Workflow:
  1. Load every repository present in
     1_configurations_artifacts_results/clone_terms_report.json.
  2. For each repository, compute the earliest artifact created_at timestamp
     found in the report (falling back to timestamp when needed).
  3. Query the GitHub Search API directly for merged pull requests where at
      least one CLONE_TERM already appears in the PR body or issue comments,
      with merged_at >= that cutoff.
  4. For every candidate PR found, fetch the PR body plus all discussion surfaces:
     issue comments, review summaries, and inline review comments.
  5. Classify each author/comment as human or agent using lightweight
     heuristics based on GitHub account metadata, login patterns, and a few
     explicit AI-generated signatures in the comment text.
  6. Record clone-term matches while preserving the complete collected
     discussion payload for each PR.

Outputs:
    - clone_term_prs_results/clone_term_prs.json
    - clone_term_prs_results/clone_term_prs_checkpoint.json

Notes:
  - This script uses the GitHub Search API, which only exposes the first 1,000
        results for a given query. Repositories above that limit are flagged as
        truncated in the output.
    - Direct GitHub issue search covers PR title/body and issue comments, but not
        PR review comments; review and inline-review comments are still fetched
        after candidate discovery.
    - Search results filtered with is:merged expose the merge timestamp through
    closed_at.
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

OUTPUT_DIR = ROOT_DIR / "clone_term_prs_results"
RESULTS_PATH = OUTPUT_DIR / "clone_term_prs.json"
CHECKPOINT_PATH = OUTPUT_DIR / "clone_term_prs_checkpoint.json"

CHECKPOINT_VERSION = 4
GITHUB_API_BASE = "https://api.github.com"
GITHUB_SEARCH_URL = f"{GITHUB_API_BASE}/search/issues"
GITHUB_REPO_API_URL = f"{GITHUB_API_BASE}/repos" + "/{repo_name}"
GITHUB_TOKEN_ENV = "GITHUB_TOKEN"
GITHUB_TOKEN_PREFIX = "GITHUB_TOKEN_"
REQUEST_TIMEOUT_SECONDS = 60
REQUEST_DELAY_SECONDS = 0.0
PER_PAGE = 100
SNIPPET_CONTEXT = 150
CHECKPOINT_SAVE_EVERY = 1
SEARCH_TEXT_MATCH_ACCEPT = "application/vnd.github.text-match+json"

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

_BOT_LOGIN_RE = re.compile(
    r"\[bot\]$|^(dependabot|renovate|github-actions|codecov|snyk|deepsource|codeclimate|mergify|allcontributors|semantic-release|release-drafter|greenkeeper|imgbot|whitesource|sonarqubebot|stale|netlify)(\[bot\])?$",
    re.IGNORECASE,
)
_AGENT_LOGIN_KEYWORDS = (
    "copilot",
    "coderabbit",
    "sweep",
    "codium",
    "qodo",
    "augment",
    "devin",
    "aider",
    "cline",
    "roo",
    "sourcegraph",
    "codeium",
)
_AGENT_TEXT_RE = re.compile(
    r"\b(generated by|automated comment|ai-generated|copilot review|coderabbit|sweep[- ]ai|codiumai|qodo|augment code|devin|openai assistant)\b",
    re.IGNORECASE,
)


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


# ---------------------------------------------------------------------------
# GitHub token pool
# ---------------------------------------------------------------------------

def base_headers() -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "dataset-code-duplication-clone-term-pr-scan/2.0",
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


def github_get(
    url: str,
    pool: GitHubTokenPool,
    *,
    params: dict[str, Any] | None = None,
    extra_headers: dict[str, str] | None = None,
) -> requests.Response:
    while True:
        headers, entry = pool.next_headers()
        if extra_headers:
            headers.update(extra_headers)
        response = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if should_retry(response, pool, entry):
            continue
        return response


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Text scanning and actor classification
# ---------------------------------------------------------------------------

def is_bot(login: str) -> bool:
    return bool(_BOT_LOGIN_RE.search(login))


def classify_actor(login: str, github_user_type: str = "", text: str = "") -> tuple[str, str]:
    normalized_login = login.strip().lower()
    normalized_type = github_user_type.strip().lower()

    if normalized_type == "bot":
        return "agent", "github_user_type=Bot"
    if is_bot(login):
        return "agent", "bot_login_pattern"
    for keyword in _AGENT_LOGIN_KEYWORDS:
        if keyword in normalized_login:
            return "agent", f"login_keyword:{keyword}"
    if text and _AGENT_TEXT_RE.search(text):
        return "agent", "agent_signature_in_text"
    return "human", "default_human"


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


def scan_text(
    text: str,
    location: str,
    author: str,
    author_type_value: str,
    author_type_reason: str,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for term in CLONE_TERMS:
        snippets = extract_snippets(text, term)
        if snippets:
            matches.append(
                {
                    "location": location,
                    "author": author,
                    "author_type": author_type_value,
                    "author_type_reason": author_type_reason,
                    "term": term,
                    "excerpts": snippets[:3],
                }
            )
    return matches


def matched_terms_from_matches(matches: list[dict[str, Any]]) -> list[str]:
    return sorted({match["term"] for match in matches})


def summarize_comments(comments: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "total_comments": len(comments),
        "human_comments": 0,
        "agent_comments": 0,
        "issue_comments": 0,
        "review_summaries": 0,
        "inline_review_comments": 0,
        "comments_with_clone_terms": 0,
    }

    for comment in comments:
        if comment.get("author_type") == "agent":
            summary["agent_comments"] += 1
        else:
            summary["human_comments"] += 1

        kind = comment.get("kind")
        if kind == "issue_comment":
            summary["issue_comments"] += 1
        elif kind == "review":
            summary["review_summaries"] += 1
        elif kind == "inline_review_comment":
            summary["inline_review_comments"] += 1

        if comment.get("contains_clone_terms"):
            summary["comments_with_clone_terms"] += 1

    return summary


# ---------------------------------------------------------------------------
# GitHub REST fetchers
# ---------------------------------------------------------------------------

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


def build_search_query(repo_name: str, merged_since: str, term: str) -> str:
    return f'repo:{repo_name} is:pr is:merged merged:>={merged_since} "{term}"'


def fetch_all_pages(url: str, pool: GitHubTokenPool, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
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


def fetch_direct_pr_candidates(repo_name: str, merged_since: str, pool: GitHubTokenPool) -> dict[str, Any]:
    candidates_by_number: dict[int, dict[str, Any]] = {}
    query_repo_name = repo_name
    total_count = 0
    term_searches: list[dict[str, Any]] = []
    truncated = False

    for term in CLONE_TERMS:
        page = 1
        term_total_count = 0
        term_truncated = False
        term_incomplete_results = False
        term_raw_matches = 0

        while True:
            params = {
                "q": build_search_query(query_repo_name, merged_since, term),
                "sort": "updated",
                "order": "desc",
                "per_page": PER_PAGE,
                "page": page,
            }

            response = github_get(
                GITHUB_SEARCH_URL,
                pool,
                params=params,
                extra_headers={"Accept": SEARCH_TEXT_MATCH_ACCEPT},
            )
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
                term_total_count = int(payload.get("total_count") or 0)
                total_count += term_total_count
                term_incomplete_results = bool(payload.get("incomplete_results"))

            items = payload.get("items") or []
            term_raw_matches += len(items)
            for item in items:
                number = int(item.get("number") or 0)
                merged_at = str(item.get("closed_at") or "").strip()
                candidate = candidates_by_number.setdefault(
                    number,
                    {
                        "number": number,
                        "title": str(item.get("title") or "").strip(),
                        "url": str(item.get("html_url") or "").strip(),
                        "merged_at": merged_at,
                        "created_at": str(item.get("created_at") or "").strip(),
                        "updated_at": str(item.get("updated_at") or "").strip(),
                        "author_login": str((item.get("user") or {}).get("login") or "").strip(),
                        "author_user_type": str((item.get("user") or {}).get("type") or "").strip(),
                        "search_matched_terms": set(),
                        "search_text_matches": [],
                        "has_issue_comment_text_match": False,
                        "has_body_text_match": False,
                    },
                )
                candidate["search_matched_terms"].add(term)

                for text_match in item.get("text_matches") or []:
                    object_type = str(text_match.get("object_type") or "").strip()
                    property_name = str(text_match.get("property") or "").strip()
                    fragment = str(text_match.get("fragment") or "").strip()
                    matched_texts = [str(match.get("text") or "").strip() for match in text_match.get("matches") or []]
                    candidate["search_text_matches"].append(
                        {
                            "term": term,
                            "object_type": object_type,
                            "object_url": str(text_match.get("object_url") or "").strip(),
                            "property": property_name,
                            "fragment": fragment,
                            "matched_texts": matched_texts,
                        }
                    )
                    if object_type.lower() == "issuecomment" and property_name == "body":
                        candidate["has_issue_comment_text_match"] = True
                    if object_type.lower() == "issue" and property_name == "body":
                        candidate["has_body_text_match"] = True

            fetched_so_far = page * PER_PAGE
            capped_total = min(term_total_count, 1000)
            if term_total_count > 1000 and fetched_so_far >= 1000:
                term_truncated = True
                truncated = True
                break
            if len(items) < PER_PAGE or fetched_so_far >= capped_total:
                break

            page += 1
            if REQUEST_DELAY_SECONDS > 0:
                time.sleep(REQUEST_DELAY_SECONDS)

        term_searches.append(
            {
                "term": term,
                "total_count": term_total_count,
                "raw_matches_collected": term_raw_matches,
                "truncated_by_search_limit": term_truncated,
                "incomplete_results": term_incomplete_results,
            }
        )

    collected = []
    for candidate in candidates_by_number.values():
        candidate["search_matched_terms"] = sorted(candidate["search_matched_terms"])
        candidate["search_text_matches"].sort(
            key=lambda item: (
                item.get("object_type") or "",
                item.get("property") or "",
                item.get("term") or "",
                item.get("object_url") or "",
            )
        )
        collected.append(candidate)

    collected.sort(key=lambda item: item.get("merged_at") or "")
    return {
        "queried_repo_name": query_repo_name,
        "search_cutoff_timestamp": merged_since,
        "total_count": total_count,
        "truncated_by_search_limit": truncated,
        "term_searches": term_searches,
        "candidate_prs": collected,
    }


def fetch_pr_details(repo: str, number: int, pool: GitHubTokenPool) -> dict[str, Any]:
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{number}"
    response = github_get(url, pool)
    response.raise_for_status()
    return response.json()


def fetch_issue_comments(repo: str, number: int, pool: GitHubTokenPool) -> list[dict[str, Any]]:
    url = f"{GITHUB_API_BASE}/repos/{repo}/issues/{number}/comments"
    return fetch_all_pages(url, pool)


def fetch_review_comments(repo: str, number: int, pool: GitHubTokenPool) -> list[dict[str, Any]]:
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{number}/reviews"
    return fetch_all_pages(url, pool)


def fetch_inline_review_comments(repo: str, number: int, pool: GitHubTokenPool) -> list[dict[str, Any]]:
    url = f"{GITHUB_API_BASE}/repos/{repo}/pulls/{number}/comments"
    return fetch_all_pages(url, pool)


def build_comment_entry(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    user = payload.get("user") or {}
    author = str(user.get("login") or "").strip()
    github_user_type = str(user.get("type") or "").strip()
    body = str(payload.get("body") or "")
    actor_type, actor_reason = classify_actor(author, github_user_type, body)
    matched_terms = matched_terms_from_matches(scan_text(body, kind, author, actor_type, actor_reason)) if body else []

    return {
        "kind": kind,
        "id": int(payload.get("id") or 0),
        "author": author,
        "author_type": actor_type,
        "author_type_reason": actor_reason,
        "github_user_type": github_user_type,
        "created_at": str(payload.get("submitted_at") or payload.get("created_at") or "").strip(),
        "url": str(payload.get("html_url") or payload.get("url") or "").strip(),
        "review_state": str(payload.get("state") or "").strip(),
        "body": body,
        "contains_clone_terms": bool(matched_terms),
        "matched_terms": matched_terms,
    }


# ---------------------------------------------------------------------------
# Per-PR scan
# ---------------------------------------------------------------------------

def scan_pr(
    repo: str,
    pr: dict[str, Any],
    pool: GitHubTokenPool,
) -> dict[str, Any]:
    number = pr["number"]

    try:
        details = fetch_pr_details(repo, number, pool)
    except Exception as exc:
        print(f"    [WARN] Could not fetch PR details for #{number}: {exc}")
        details = {}

    author = str((details.get("user") or {}).get("login") or pr.get("author_login") or "").strip()
    author_user_type = str((details.get("user") or {}).get("type") or pr.get("author_user_type") or "").strip()
    pr_author_type, pr_author_reason = classify_actor(author, author_user_type)

    body = str(details.get("body") or "")
    body_matches = scan_text(body, "body", author, pr_author_type, pr_author_reason) if body else []
    all_matches = list(body_matches)
    comment_matches: list[dict[str, Any]] = []

    try:
        issue_comments = fetch_issue_comments(repo, number, pool)
    except Exception as exc:
        print(f"    [WARN] Could not fetch issue comments for #{number}: {exc}")
        issue_comments = []

    try:
        reviews = fetch_review_comments(repo, number, pool)
    except Exception as exc:
        print(f"    [WARN] Could not fetch reviews for #{number}: {exc}")
        reviews = []

    try:
        inline_review_comments = fetch_inline_review_comments(repo, number, pool)
    except Exception as exc:
        print(f"    [WARN] Could not fetch inline review comments for #{number}: {exc}")
        inline_review_comments = []

    comments: list[dict[str, Any]] = []
    for kind, payloads in (
        ("issue_comment", issue_comments),
        ("review", reviews),
        ("inline_review_comment", inline_review_comments),
    ):
        for payload in payloads:
            comment_entry = build_comment_entry(kind, payload)
            comments.append(comment_entry)
            if comment_entry["body"]:
                current_comment_matches = scan_text(
                    comment_entry["body"],
                    kind,
                    comment_entry["author"],
                    comment_entry["author_type"],
                    comment_entry["author_type_reason"],
                )
                comment_matches.extend(current_comment_matches)
                all_matches.extend(current_comment_matches)

    comments.sort(key=lambda item: (item.get("created_at") or "", item.get("kind") or "", item.get("id") or 0))
    matched_terms = matched_terms_from_matches(all_matches)
    comment_matched_terms = matched_terms_from_matches(comment_matches)
    has_human_match = any(match["author_type"] == "human" for match in all_matches)
    has_human_comment_match = any(match["author_type"] == "human" for match in comment_matches)

    return {
        "number": number,
        "title": str(details.get("title") or pr.get("title") or "").strip(),
        "url": str(details.get("html_url") or pr.get("url") or "").strip(),
        "merged_at": str(details.get("merged_at") or pr.get("merged_at") or "").strip(),
        "created_at": str(details.get("created_at") or pr.get("created_at") or "").strip(),
        "updated_at": str(details.get("updated_at") or pr.get("updated_at") or "").strip(),
        "author_login": author,
        "author_type": pr_author_type,
        "author_type_reason": pr_author_reason,
        "github_user_type": author_user_type,
        "contains_body_clone_terms": bool(body_matches),
        "contains_comment_clone_terms": bool(comment_matches),
        "contains_clone_terms": bool(all_matches),
        "matched_terms": matched_terms,
        "comment_matched_terms": comment_matched_terms,
        "has_human_match": has_human_match,
        "has_human_comment_match": has_human_comment_match,
        "matches": all_matches,
        "comment_matches": comment_matches,
        "search_matched_terms": pr.get("search_matched_terms", []),
        "search_text_matches": pr.get("search_text_matches", []),
        "has_issue_comment_text_match": bool(pr.get("has_issue_comment_text_match")),
        "has_body_text_match": bool(pr.get("has_body_text_match")),
        "comments": comments,
        "comment_summary": summarize_comments(comments),
    }


# ---------------------------------------------------------------------------
# Per-repo processing
# ---------------------------------------------------------------------------

def process_repo(
    repo: str,
    repo_target: dict[str, Any],
    pool: GitHubTokenPool,
) -> dict[str, Any]:
    fetch_result = fetch_direct_pr_candidates(repo, repo_target["cutoff_created_at"], pool)
    candidate_prs = fetch_result["candidate_prs"]
    scanned_prs: list[dict[str, Any]] = []

    print(f"  Searching merged PRs with direct term matches since {repo_target['cutoff_created_at']}...")
    print(f"  Scanning {len(candidate_prs)} candidate PRs returned by GitHub search...")

    for scanned, pr in enumerate(candidate_prs, start=1):
        result = scan_pr(fetch_result["queried_repo_name"], pr, pool)
        scanned_prs.append(result)

        if result["contains_comment_clone_terms"]:
            terms_str = ", ".join(f'"{term}"' for term in result["comment_matched_terms"])
            human_flag = " [human]" if result["has_human_comment_match"] else " [agent only]"
            print(f"    #{result['number']} MATCH{human_flag}: {terms_str}")

        if scanned % 25 == 0:
            clone_matches = sum(1 for item in scanned_prs if item["contains_comment_clone_terms"])
            print(f"    ... {scanned}/{len(candidate_prs)} scanned, {clone_matches} comment clone-term PR(s) so far")

    clone_term_prs = [pr for pr in scanned_prs if pr["contains_comment_clone_terms"]]
    total_human_match_prs = sum(1 for pr in clone_term_prs if pr["has_human_comment_match"])
    total_comments = sum(pr["comment_summary"]["total_comments"] for pr in scanned_prs)
    total_human_comments = sum(pr["comment_summary"]["human_comments"] for pr in scanned_prs)
    total_agent_comments = sum(pr["comment_summary"]["agent_comments"] for pr in scanned_prs)

    return {
        "status": "completed",
        "queried_repo_name": fetch_result["queried_repo_name"],
        "search_cutoff_timestamp": repo_target["cutoff_created_at"],
        "artifact_count": repo_target["artifact_count"],
        "artifact_matches": repo_target["artifact_matches"],
        "truncated_by_search_limit": fetch_result["truncated_by_search_limit"],
        "search_total_count": fetch_result["total_count"],
        "term_searches": fetch_result["term_searches"],
        "total_prs_scanned": len(scanned_prs),
        "total_clone_term_prs": len(clone_term_prs),
        "total_human_match_prs": total_human_match_prs,
        "total_comments_collected": total_comments,
        "total_human_comments": total_human_comments,
        "total_agent_comments": total_agent_comments,
        "prs": clone_term_prs,
    }


# ---------------------------------------------------------------------------
# Checkpoint / results
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": CHECKPOINT_VERSION, "repos": {}}

    payload = load_json(path)
    if payload.get("version") != CHECKPOINT_VERSION:
        return {"version": CHECKPOINT_VERSION, "repos": {}}
    return payload


def build_summary(repos_state: dict[str, Any], repo_names: list[str]) -> dict[str, Any]:
    completed = [r for r in repo_names if repos_state.get(r, {}).get("status") == "completed"]
    failed = [r for r in repo_names if repos_state.get(r, {}).get("status") == "failed"]
    truncated = [r for r in completed if repos_state.get(r, {}).get("truncated_by_search_limit")]
    total_prs = sum(repos_state[r].get("total_clone_term_prs", 0) for r in completed)
    total_human = sum(repos_state[r].get("total_human_match_prs", 0) for r in completed)
    total_scanned = sum(repos_state[r].get("total_prs_scanned", 0) for r in completed)
    total_comments = sum(repos_state[r].get("total_comments_collected", 0) for r in completed)
    total_human_comments = sum(repos_state[r].get("total_human_comments", 0) for r in completed)
    total_agent_comments = sum(repos_state[r].get("total_agent_comments", 0) for r in completed)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_configuration_report_path": str(CONFIG_REPORT_PATH),
        "clone_terms_searched": CLONE_TERMS,
        "total_repositories": len(repo_names),
        "repositories_completed": len(completed),
        "repositories_failed": len(failed),
        "repositories_truncated_by_search_limit": truncated,
        "total_prs_scanned": total_scanned,
        "total_clone_term_prs": total_prs,
        "total_human_match_prs": total_human,
        "total_comments_collected": total_comments,
        "total_human_comments": total_human_comments,
        "total_agent_comments": total_agent_comments,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            print(
                f"  -> {result['total_clone_term_prs']} clone-term PR(s); "
                f"{result['total_comments_collected']} comment(s) collected"
            )
        except Exception as exc:
            repos_state[repo] = {
                "status": "failed",
                "search_cutoff_timestamp": repo_target["cutoff_created_at"],
                "error": str(exc),
                "prs": [],
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
    print(f"Repositories truncated:  {len(summary['repositories_truncated_by_search_limit'])}")
    print(f"Total PRs scanned:       {summary['total_prs_scanned']}")
    print(f"Total clone-term PRs:    {summary['total_clone_term_prs']}")
    print(f"Total comments captured: {summary['total_comments_collected']}")
    print(f"  Human comments:        {summary['total_human_comments']}")
    print(f"  Agent comments:        {summary['total_agent_comments']}")
    print(f"Results:   {RESULTS_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()