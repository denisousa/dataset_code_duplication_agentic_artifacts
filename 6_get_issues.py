import json
import os
import re
import threading
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv

load_dotenv()


class TokenPool:
    """Round-robin pool of GitHub personal access tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = [token for token in tokens if token]
        self._index = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def current(self) -> str | None:
        return self._tokens[self._index] if self._tokens else None

    def rotate(self) -> str | None:
        """Advance to the next token (thread-safe, wraps around)."""
        if not self._tokens:
            return None
        with self._lock:
            self._index = (self._index + 1) % len(self._tokens)
            return self._tokens[self._index]


def _load_token_pool() -> TokenPool:
    """Load tokens from GITHUB_TOKEN_1 … GITHUB_TOKEN_9, falling back to GITHUB_TOKEN."""
    tokens: list[str] = []
    for index in range(1, 10):
        token = os.environ.get(f"GITHUB_TOKEN_{index}", "").strip()
        if token:
            tokens.append(token)
    if not tokens:
        single = os.environ.get("GITHUB_TOKEN", "").strip()
        if single:
            tokens.append(single)
    return TokenPool(tokens)


_token_pool: TokenPool = _load_token_pool()

RESULTS_DIR = Path("6_issues_results")
REPORT_PATH = Path("1_context_results/clone_terms_report.json")
CHECKPOINT_PATH = RESULTS_DIR / "search_issues_checkpoint.json"
OUTPUT_PATH = RESULTS_DIR / "search_issues_results.json"

CLONE_TERMS: list[str] = [
    "duplicate code",
    "duplicated code",
    "code duplication",
    "repeated code",
    "DRY principle",
    "avoid duplication",
    "reused code",
    "code reuse",
    "code sharing",
    "don't repeat yourself",
    "redundant code",
    "duplicate functionality",
    "duplicated functionality",
]

AGENT_MARKERS: tuple[str, ...] = (
    "claude",
    "copilot",
    "cursor",
    "gemini",
    "codex",
)
PATTERN = re.compile("|".join(CLONE_TERMS), re.IGNORECASE)
AGENT_MARKER_RE = re.compile("|".join(re.escape(marker) for marker in AGENT_MARKERS), re.IGNORECASE)
BASE_URL = "https://api.github.com"

MAX_ISSUE_WORKERS = 5
MAX_REPO_WORKERS = 2

PRINT_LOCK = threading.Lock()


def print_repo_summary(repo_full: str, results: List[Dict[str, Any]]) -> None:
    """Print all valid issues for a repo as soon as it finishes (thread-safe)."""
    with PRINT_LOCK:
        print(f"\n{'#' * 80}")
        if not results:
            print(f"DONE  {repo_full}  – no matches found")
        else:
            print(f"DONE  {repo_full}  – {len(results)} matching issue(s)")
            for result in sorted(results, key=lambda item: item["closed_at"] or ""):
                disc = result["discussion_type"].upper()
                actor = result.get("actor_type", "unknown").upper()
                commenter_types = result.get("commenter_types") or "—"
                terms = ", ".join(result.get("matched_terms", []))
                matched_comments_count = result.get("matched_comments_count", 0)
                print(f"  Issue #{result['number']} [{disc}] [{actor}] – {result['title']}")
                print(f"     Closed at            : {result['closed_at']}")
                print(f"     URL                  : {result['url']}")
                print(f"     Terms matched        : {terms}")
                print(f"     Comments with term   : {matched_comments_count}")
                print(f"     Commenter types      : {commenter_types}")
                print("     Matches:")
                for match in result["matches"]:
                    author_tag = f" @{match['author']} [{match['author_type']}]" if match["author"] else ""
                    print(f"       [{match['location']}{author_tag}] '{match['term']}' → {match['excerpt']}")
        print(f"{'#' * 80}")


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=50)
    session.mount("https://", adapter)
    headers: Dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = _token_pool.current
    if token:
        headers["Authorization"] = f"Bearer {token}"
        print(f"  Token pool: {len(_token_pool)} token(s) loaded.")
    else:
        print("[WARNING] No GITHUB_TOKEN found – unauthenticated requests are severely rate-limited.")
    session.headers.update(headers)
    return session


SESSION = _make_session()


def _empty_checkpoint() -> Dict[str, Any]:
    return {
        "version": 2,
        "repos": {},
    }


def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint data, upgrading legacy repo->results checkpoints on the fly."""
    if not CHECKPOINT_PATH.exists():
        return _empty_checkpoint()

    with CHECKPOINT_PATH.open(encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    if isinstance(data, dict) and data.get("version") == 2 and isinstance(data.get("repos"), dict):
        return data

    repos: Dict[str, Any] = {}
    if isinstance(data, dict):
        for repo_full, results in data.items():
            if not isinstance(repo_full, str):
                continue
            repo_results = results if isinstance(results, list) else []
            processed_issues = {
                str(result.get("number")): result
                for result in repo_results
                if isinstance(result, dict) and result.get("number") is not None
            }
            repos[repo_full] = {
                "status": "completed",
                "results": repo_results,
                "processed_issues": processed_issues,
            }

    return {
        "version": 2,
        "repos": repos,
    }


def save_checkpoint(data: Dict[str, Any], lock: threading.Lock) -> None:
    """Atomically write the checkpoint file (write-then-rename)."""
    with lock:
        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CHECKPOINT_PATH)


def save_results(all_results: Dict[str, List[Dict[str, Any]]], lock: threading.Lock) -> None:
    """Atomically write the final results file."""
    with lock:
        tmp = OUTPUT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(OUTPUT_PATH)


def load_repos_from_report() -> List[str]:
    """Return unique repo names (owner/repo) from the clone terms report."""
    with REPORT_PATH.open(encoding="utf-8") as file_handle:
        report = json.load(file_handle)

    seen: set[str] = set()
    for source in report.get("sources", []):
        for match in source.get("matches", []):
            repo_name = match.get("repo_name", "").strip()
            if repo_name:
                seen.add(repo_name)
    return sorted(seen)


def checkpoint_completed_results(checkpoint: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    completed: Dict[str, List[Dict[str, Any]]] = {}
    for repo_full, repo_state in checkpoint.get("repos", {}).items():
        if repo_state.get("status") != "completed":
            continue
        results = repo_state.get("results") or []
        if results:
            completed[repo_full] = results
    return completed


_RATE_LIMIT_MSGS = ("rate limit", "secondary rate", "api rate limit", "exceeded a secondary")


def _get_with_rate_limit_retry(
    url: str,
    params: Dict[str, Any] | None = None,
    max_attempts: int = 6,
) -> requests.Response:
    """GET *url*, sleeping and retrying automatically on rate-limit 403/429."""
    for attempt in range(max_attempts):
        response = SESSION.get(url, params=params)

        if response.status_code not in (403, 429):
            return response

        remaining = int(response.headers.get("X-RateLimit-Remaining", "1"))
        reset_ts = int(response.headers.get("X-RateLimit-Reset", "0"))
        try:
            message = response.json().get("message", "").lower()
        except Exception:
            message = ""

        is_rate_limit = (
            response.status_code == 429
            or remaining == 0
            or any(fragment in message for fragment in _RATE_LIMIT_MSGS)
        )

        if not is_rate_limit:
            return response

        if len(_token_pool) > 1:
            new_token = _token_pool.rotate()
            SESSION.headers["Authorization"] = f"Bearer {new_token}"
            with PRINT_LOCK:
                print(
                    f"\n  [RATE LIMIT] HTTP {response.status_code} – "
                    f"rotating to next token (attempt {attempt + 1}/{max_attempts}) …"
                )
            continue

        if reset_ts:
            wait = max(reset_ts - int(time.time()) + 5, 10)
        else:
            wait = min(30 * 2 ** attempt, 600)

        with PRINT_LOCK:
            print(
                f"\n  [RATE LIMIT] HTTP {response.status_code} – "
                f"waiting {wait}s (attempt {attempt + 1}/{max_attempts}) …"
            )
        time.sleep(wait)

    return response


def github_get(url: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    results = []
    while url:
        response = _get_with_rate_limit_retry(url, params)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            results.extend(data)
        else:
            results.append(data)
        url = response.links.get("next", {}).get("url")
        params = None
    return results


def find_matches_in_text(
    location: str,
    text: str | None,
    author: str | None = None,
    author_type: str | None = None,
) -> List[Dict[str, Any]]:
    """Return structured match entries for every term hit in *text*."""
    if not text:
        return []
    matches = []
    for match in PATTERN.finditer(text):
        start = max(match.start() - 50, 0)
        end = min(match.end() + 50, len(text))
        excerpt = text[start:end].replace("\n", " ")
        matches.append(
            {
                "location": location,
                "author": author,
                "author_type": author_type,
                "term": match.group(0).lower(),
                "excerpt": f"...{excerpt}...",
            }
        )
    return matches


def get_author_type(user: Dict[str, Any] | None) -> str:
    """Classify a GitHub user as 'bot' or 'human'."""
    if not user:
        return "unknown"
    if user.get("type") == "Bot":
        return "bot"
    login = user.get("login", "")
    if "[bot]" in login or login.endswith("-bot"):
        return "bot"
    return "human"


def get_closed_issues(owner: str, repo: str) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues"
    all_items = github_get(
        url,
        params={
            "state": "closed",
            "per_page": 100,
            "sort": "updated",
            "direction": "desc",
        },
    )
    return [item for item in all_items if "pull_request" not in item]


def get_issue_comments(owner: str, repo: str, issue_number: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{issue_number}/comments"
    return github_get(url, params={"per_page": 100})


def _text_has_agent_marker(text: str | None) -> bool:
    return bool(text and AGENT_MARKER_RE.search(text))


def _user_looks_agent(user: Dict[str, Any] | None) -> bool:
    if not user:
        return False
    if get_author_type(user) == "bot":
        return True
    login = user.get("login", "")
    return _text_has_agent_marker(login)


def classify_issue_actor(issue: Dict[str, Any]) -> str:
    """Classify the issue origin as 'agent' or 'human' using GitHub-visible heuristics."""
    if _user_looks_agent(issue.get("user")):
        return "agent"

    if _text_has_agent_marker(issue.get("title")) or _text_has_agent_marker(issue.get("body")):
        return "agent"

    return "human"


def analyze_issue(owner: str, repo: str, issue: Dict[str, Any]) -> Dict[str, Any] | None:
    issue_number = issue["number"]
    all_matches: List[Dict[str, Any]] = []

    all_matches += find_matches_in_text("Issue title", issue.get("title"))
    all_matches += find_matches_in_text("Issue body", issue.get("body"))

    issue_comments = get_issue_comments(owner, repo, issue_number)
    matched_comments_count = 0

    for comment in issue_comments:
        user = comment.get("user") or {}
        hits = find_matches_in_text(
            "Issue comment",
            comment.get("body"),
            author=user.get("login"),
            author_type=get_author_type(user),
        )
        if hits:
            matched_comments_count += 1
            all_matches += hits

    if not all_matches:
        return None

    has_primary = any(match["location"] == "Issue title" for match in all_matches)
    discussion_type = "dedicated" if has_primary else "mention"
    actor_type = classify_issue_actor(issue)
    matched_terms = sorted({match["term"] for match in all_matches})

    comment_author_types = {
        match["author_type"]
        for match in all_matches
        if match["location"] != "Issue title" and match["author_type"]
    }
    if "human" in comment_author_types and "bot" in comment_author_types:
        commenter_types: str | None = "both"
    elif "human" in comment_author_types:
        commenter_types = "human"
    elif "bot" in comment_author_types:
        commenter_types = "bot"
    else:
        commenter_types = None

    return {
        "number": issue_number,
        "title": issue.get("title"),
        "url": issue.get("html_url"),
        "closed_at": issue.get("closed_at"),
        "discussion_type": discussion_type,
        "actor_type": actor_type,
        "matched_terms": matched_terms,
        "matched_comments_count": matched_comments_count,
        "commenter_types": commenter_types,
        "matches": all_matches,
    }


def process_repo(
    repo_full: str,
    checkpoint: Dict[str, Any],
    ck_lock: threading.Lock,
    all_results: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, List[Dict[str, Any]]]:
    owner, repo = repo_full.split("/", 1)
    print(f"\n[{repo_full}] Fetching closed issues ...")
    try:
        issues = get_closed_issues(owner, repo)
    except requests.HTTPError as exc:
        print(f"  [ERROR] {exc}")
        return repo_full, []

    repo_state = checkpoint.setdefault("repos", {}).setdefault(
        repo_full,
        {
            "status": "in_progress",
            "results": [],
            "processed_issues": {},
        },
    )
    repo_state.setdefault("results", [])
    repo_state.setdefault("processed_issues", {})
    processed_issues: Dict[str, Any] = repo_state["processed_issues"]
    repo_state["status"] = "in_progress"
    save_checkpoint(checkpoint, ck_lock)

    total = len(issues)
    done = len(processed_issues)
    pending_issues = [issue for issue in issues if str(issue["number"]) not in processed_issues]
    results: List[Dict[str, Any]] = [
        result for result in processed_issues.values() if result is not None
    ]

    print(
        f"  {total} closed issues found – "
        f"{done} restored from checkpoint, {len(pending_issues)} pending ..."
    )

    if not pending_issues:
        repo_state["results"] = results
        repo_state["status"] = "completed"
        save_checkpoint(checkpoint, ck_lock)
        if results:
            all_results[repo_full] = results
            save_results(all_results, ck_lock)
        else:
            all_results.pop(repo_full, None)
        print_repo_summary(repo_full, results)
        return repo_full, results

    with ThreadPoolExecutor(max_workers=MAX_ISSUE_WORKERS) as executor:
        futures = {executor.submit(analyze_issue, owner, repo, issue): issue["number"] for issue in pending_issues}
        for future in as_completed(futures):
            done += 1
            with PRINT_LOCK:
                print(f"  [{done}/{total}] Issue #{futures[future]} analyzed", end="\r")
            try:
                result = future.result()
                issue_number = str(futures[future])
                if result:
                    results.append(result)
                    processed_issues[issue_number] = result
                else:
                    processed_issues[issue_number] = None
                repo_state["results"] = results
                save_checkpoint(checkpoint, ck_lock)
                if results:
                    all_results[repo_full] = results
                    save_results(all_results, ck_lock)
            except Exception as exc:
                with PRINT_LOCK:
                    print(f"\n  [WARN] Issue #{futures[future]}: {exc}")

    repo_state["results"] = results
    repo_state["status"] = "completed"
    save_checkpoint(checkpoint, ck_lock)
    if results:
        all_results[repo_full] = results
        save_results(all_results, ck_lock)
    else:
        all_results.pop(repo_full, None)

    print_repo_summary(repo_full, results)
    return repo_full, results


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    repos = [repo for repo in load_repos_from_report() if "/" in repo]
    print(f"Repositories with code-clone config files: {len(repos)}")

    checkpoint = load_checkpoint()
    repo_states: Dict[str, Any] = checkpoint.setdefault("repos", {})

    pending = [repo for repo in repos if repo_states.get(repo, {}).get("status") != "completed"]
    completed_count = sum(1 for repo in repos if repo_states.get(repo, {}).get("status") == "completed")
    resumable_count = sum(1 for repo in repos if repo_states.get(repo, {}).get("status") == "in_progress")
    if repo_states:
        print(
            f"  Checkpoint loaded – {completed_count} repo(s) completed, "
            f"{resumable_count} in progress, {len(pending)} remaining."
        )

    all_results: Dict[str, List[Dict[str, Any]]] = checkpoint_completed_results(checkpoint)
    ck_lock = threading.Lock()

    def _process_and_checkpoint(repo_full: str) -> Tuple[str, List[Dict[str, Any]]]:
        return process_repo(repo_full, checkpoint, ck_lock, all_results)

    with ThreadPoolExecutor(max_workers=MAX_REPO_WORKERS) as executor:
        futures = {executor.submit(_process_and_checkpoint, repo): repo for repo in pending}
        for future in as_completed(futures):
            future.result()

    if not all_results:
        print("No closed issues found discussing code cloning or duplication.")
        return

    total = sum(len(results) for results in all_results.values())
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Total: {total} matching issue(s) across {len(all_results)} repo(s).")


if __name__ == "__main__":
    main()