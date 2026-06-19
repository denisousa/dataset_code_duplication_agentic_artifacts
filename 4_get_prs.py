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

# ---------------------------------------------------------------------------
# Token pool – rotate across multiple GITHUB_TOKEN_N env vars on rate-limit
# ---------------------------------------------------------------------------

class TokenPool:
    """Round-robin pool of GitHub personal access tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = [t for t in tokens if t]
        self._index  = 0
        self._lock   = threading.Lock()

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
    for i in range(1, 10):
        t = os.environ.get(f"GITHUB_TOKEN_{i}", "").strip()
        if t:
            tokens.append(t)
    if not tokens:
        single = os.environ.get("GITHUB_TOKEN", "").strip()
        if single:
            tokens.append(single)
    return TokenPool(tokens)


_token_pool: TokenPool = _load_token_pool()

RESULTS_DIR      = Path("4_prs_results")
REPORT_PATH      = Path("1_context_results/clone_terms_report.json")
CHECKPOINT_PATH  = RESULTS_DIR / "search_prs_checkpoint.json"
OUTPUT_PATH      = RESULTS_DIR / "search_prs_results.json"

# English-only terms for code cloning / duplication discussions
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

# Parallelism knobs – kept conservative to avoid GitHub secondary rate limits
MAX_PR_WORKERS   = 5  # PRs analyzed concurrently per repo
MAX_REPO_WORKERS = 2  # repos processed concurrently

# Shared lock so concurrent threads don't interleave their output
PRINT_LOCK = threading.Lock()


def print_repo_summary(repo_full: str, results: List[Dict[str, Any]]) -> None:
    """Print all valid PRs for a repo as soon as it finishes (thread-safe)."""
    with PRINT_LOCK:
        print(f"\n{'#' * 80}")
        if not results:
            print(f"DONE  {repo_full}  – no matches found")
        else:
            print(f"DONE  {repo_full}  – {len(results)} matching PR(s)")
            for result in sorted(results, key=lambda r: r["merged_at"] or ""):
                disc  = result["discussion_type"].upper()
                actor = result.get("actor_type", "unknown").upper()
                ct    = result.get("commenter_types") or "—"
                terms = ", ".join(result.get("matched_terms", []))
                n_comments = result.get("matched_comments_count", 0)
                print(f"  PR #{result['number']} [{disc}] [{actor}] – {result['title']}")
                print(f"     Merged at             : {result['merged_at']}")
                print(f"     URL                   : {result['url']}")
                print(f"     Terms matched         : {terms}")
                print(f"     Comments with term    : {n_comments}")
                print(f"     Commenter types       : {ct}")
                print(f"     Matches:")
                for m in result["matches"]:
                    author_tag = f" @{m['author']} [{m['author_type']}]" if m["author"] else ""
                    print(f"       [{m['location']}{author_tag}] '{m['term']}' → {m['excerpt']}")
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


# Single shared session – connection pool is reused across threads
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

    with CHECKPOINT_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and data.get("version") == 2 and isinstance(data.get("repos"), dict):
        return data

    repos: Dict[str, Any] = {}
    if isinstance(data, dict):
        for repo_full, results in data.items():
            if not isinstance(repo_full, str):
                continue
            repo_results = results if isinstance(results, list) else []
            processed_prs = {
                str(result.get("number")): result
                for result in repo_results
                if isinstance(result, dict) and result.get("number") is not None
            }
            repos[repo_full] = {
                "status": "completed",
                "results": repo_results,
                "processed_prs": processed_prs,
            }

    return {
        "version": 2,
        "repos": repos,
    }


def save_checkpoint(
    data: Dict[str, Any],
    lock: threading.Lock,
) -> None:
    """Atomically write the checkpoint file (write-then-rename)."""
    with lock:
        tmp = CHECKPOINT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CHECKPOINT_PATH)


def save_results(
    all_results: Dict[str, List[Dict[str, Any]]],
    lock: threading.Lock,
) -> None:
    """Atomically write the final results file."""
    with lock:
        tmp = OUTPUT_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(OUTPUT_PATH)


def load_repos_from_report() -> List[str]:
    """Return unique repo names (owner/repo) from the clone terms report."""
    with REPORT_PATH.open(encoding="utf-8") as f:
        report = json.load(f)

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
            return response  # success or non-rate-limit error – let caller handle

        # Decide if this is really a rate-limit response
        remaining  = int(response.headers.get("X-RateLimit-Remaining", "1"))
        reset_ts   = int(response.headers.get("X-RateLimit-Reset", "0"))
        try:
            message = response.json().get("message", "").lower()
        except Exception:
            message = ""

        is_rate_limit = (
            response.status_code == 429
            or remaining == 0
            or any(s in message for s in _RATE_LIMIT_MSGS)
        )

        if not is_rate_limit:
            return response  # genuine 403 Forbidden – caller will raise_for_status

        # Rotate to next token if pool has more than one
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
            wait = min(30 * 2 ** attempt, 600)  # exponential: 30s, 60s, 120s … max 10 min

        with PRINT_LOCK:
            print(
                f"\n  [RATE LIMIT] HTTP {response.status_code} – "
                f"waiting {wait}s (attempt {attempt + 1}/{max_attempts}) …"
            )
        time.sleep(wait)

    return response  # return last response after exhausting retries


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
    for m in PATTERN.finditer(text):
        start = max(m.start() - 50, 0)
        end = min(m.end() + 50, len(text))
        excerpt = text[start:end].replace("\n", " ")
        matches.append({
            "location":    location,
            "author":      author,
            "author_type": author_type,
            "term":        m.group(0).lower(),
            "excerpt":     f"...{excerpt}...",
        })
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


def get_merged_pull_requests(owner: str, repo: str) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/repos/{owner}/{repo}/pulls"
    all_prs = github_get(url, params={
        "state": "closed",
        "per_page": 100,
        "sort": "updated",
        "direction": "desc",
    })
    return [pr for pr in all_prs if pr.get("merged_at") is not None]


def get_issue_comments(owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/repos/{owner}/{repo}/issues/{pr_number}/comments"
    return github_get(url, params={"per_page": 100})


def get_review_comments(owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
    return github_get(url, params={"per_page": 100})


def get_reviews(owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
    return github_get(url, params={"per_page": 100})


def get_pr_commits(owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
    url = f"{BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}/commits"
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


def classify_pr_actor(pr: Dict[str, Any], commits: List[Dict[str, Any]]) -> str:
    """Classify the PR origin as 'agent' or 'human' using GitHub-visible heuristics."""
    if _user_looks_agent(pr.get("user")):
        return "agent"

    if _text_has_agent_marker(pr.get("title")) or _text_has_agent_marker(pr.get("body")):
        return "agent"

    for commit in commits:
        author = commit.get("author") or {}
        committer = commit.get("committer") or {}
        commit_info = commit.get("commit") or {}
        commit_author = commit_info.get("author") or {}
        commit_committer = commit_info.get("committer") or {}
        message = commit_info.get("message") or ""

        if _user_looks_agent(author) or _user_looks_agent(committer):
            return "agent"

        if _text_has_agent_marker(commit_author.get("name")):
            return "agent"

        if _text_has_agent_marker(commit_committer.get("name")):
            return "agent"

        if _text_has_agent_marker(message):
            return "agent"

    return "human"


def analyze_pr(owner: str, repo: str, pr: Dict[str, Any]) -> Dict[str, Any] | None:
    pr_number = pr["number"]
    all_matches: List[Dict[str, Any]] = []

    # ── Primary sources (title + opening body) ──────────────────────────────
    all_matches += find_matches_in_text("PR title", pr.get("title"))
    all_matches += find_matches_in_text("PR body",  pr.get("body"))

    # ── Fetch discussion sources and PR commits in parallel ──────────────────
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_issue   = ex.submit(get_issue_comments,  owner, repo, pr_number)
        f_review  = ex.submit(get_review_comments, owner, repo, pr_number)
        f_reviews = ex.submit(get_reviews,          owner, repo, pr_number)
        f_commits = ex.submit(get_pr_commits,       owner, repo, pr_number)
        issue_comments  = f_issue.result()
        review_comments = f_review.result()
        reviews         = f_reviews.result()
        commits         = f_commits.result()

    matched_comments_count = 0

    for comment in issue_comments:
        user = comment.get("user") or {}
        hits = find_matches_in_text(
            "General comment",
            comment.get("body"),
            author=user.get("login"),
            author_type=get_author_type(user),
        )
        if hits:
            matched_comments_count += 1
            all_matches += hits
    for comment in review_comments:
        user = comment.get("user") or {}
        hits = find_matches_in_text(
            "Review comment",
            comment.get("body"),
            author=user.get("login"),
            author_type=get_author_type(user),
        )
        if hits:
            matched_comments_count += 1
            all_matches += hits
    for review in reviews:
        user = review.get("user") or {}
        hits = find_matches_in_text(
            "Review",
            review.get("body"),
            author=user.get("login"),
            author_type=get_author_type(user),
        )
        if hits:
            matched_comments_count += 1
            all_matches += hits

    if not all_matches:
        return None

    # ── Discussion type: dedicated / mention ─────────────────────────────────
    # dedicated = term appears in the PR title (regardless of other locations).
    # mention   = term does NOT appear in the PR title.
    has_primary = any(m["location"] == "PR title" for m in all_matches)
    discussion_type = "dedicated" if has_primary else "mention"
    actor_type = classify_pr_actor(pr, commits)

    # ── Unique matched terms ──────────────────────────────────────────────────
    matched_terms = sorted({m["term"] for m in all_matches})

    # ── Commenter types (secondary sources = everything except PR title) ──────
    comment_author_types = {
        m["author_type"]
        for m in all_matches
        if m["location"] != "PR title" and m["author_type"]
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
        "number":                 pr_number,
        "title":                  pr.get("title"),
        "url":                    pr.get("html_url"),
        "merged_at":              pr.get("merged_at"),
        "discussion_type":        discussion_type,
        "actor_type":             actor_type,
        "matched_terms":          matched_terms,
        "matched_comments_count": matched_comments_count,
        "commenter_types":        commenter_types,
        "matches":                all_matches,
    }


def process_repo(
    repo_full: str,
    checkpoint: Dict[str, Any],
    ck_lock: threading.Lock,
    all_results: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, List[Dict[str, Any]]]:
    owner, repo = repo_full.split("/", 1)
    print(f"\n[{repo_full}] Fetching merged PRs ...")
    try:
        prs = get_merged_pull_requests(owner, repo)
    except requests.HTTPError as exc:
        print(f"  [ERROR] {exc}")
        return repo_full, []

    repo_state = checkpoint.setdefault("repos", {}).setdefault(
        repo_full,
        {
            "status": "in_progress",
            "results": [],
            "processed_prs": {},
        },
    )
    repo_state.setdefault("results", [])
    repo_state.setdefault("processed_prs", {})
    processed_prs: Dict[str, Any] = repo_state["processed_prs"]
    repo_state["status"] = "in_progress"
    save_checkpoint(checkpoint, ck_lock)

    total = len(prs)
    done = len(processed_prs)
    pending_prs = [pr for pr in prs if str(pr["number"]) not in processed_prs]
    results: List[Dict[str, Any]] = [
        result for result in processed_prs.values() if result is not None
    ]

    print(
        f"  {total} merged PRs found – "
        f"{done} restored from checkpoint, {len(pending_prs)} pending ..."
    )

    if not pending_prs:
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

    with ThreadPoolExecutor(max_workers=MAX_PR_WORKERS) as ex:
        futures = {ex.submit(analyze_pr, owner, repo, pr): pr["number"] for pr in pending_prs}
        for fut in as_completed(futures):
            done += 1
            with PRINT_LOCK:
                print(f"  [{done}/{total}] PR #{futures[fut]} analyzed", end="\r")
            try:
                result = fut.result()
                pr_number = str(futures[fut])
                if result:
                    results.append(result)
                    processed_prs[pr_number] = result
                else:
                    processed_prs[pr_number] = None
                repo_state["results"] = results
                save_checkpoint(checkpoint, ck_lock)
                if results:
                    all_results[repo_full] = results
                    save_results(all_results, ck_lock)
            except Exception as exc:
                with PRINT_LOCK:
                    print(f"\n  [WARN] PR #{futures[fut]}: {exc}")

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


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    repos = [r for r in load_repos_from_report() if "/" in r]
    print(f"Repositories with code-clone config files: {len(repos)}")

    checkpoint = load_checkpoint()
    repo_states: Dict[str, Any] = checkpoint.setdefault("repos", {})
    pending = [r for r in repos if repo_states.get(r, {}).get("status") != "completed"]
    completed_count = sum(1 for r in repos if repo_states.get(r, {}).get("status") == "completed")
    resumable_count = sum(1 for r in repos if repo_states.get(r, {}).get("status") == "in_progress")
    if repo_states:
        print(
            f"  Checkpoint loaded – {completed_count} repo(s) completed, "
            f"{resumable_count} in progress, {len(pending)} remaining."
        )

    # all_results starts populated with previously saved matches
    all_results: Dict[str, List[Dict[str, Any]]] = checkpoint_completed_results(checkpoint)

    ck_lock = threading.Lock()

    def _process_and_checkpoint(repo_full: str) -> Tuple[str, List[Dict[str, Any]]]:
        return process_repo(repo_full, checkpoint, ck_lock, all_results)

    with ThreadPoolExecutor(max_workers=MAX_REPO_WORKERS) as ex:
        futures = {ex.submit(_process_and_checkpoint, r): r for r in pending}
        for fut in as_completed(futures):
            fut.result()  # surface exceptions

    if not all_results:
        print("No merged PRs found discussing code cloning or duplication.")
        return

    total = sum(len(v) for v in all_results.values())
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Total: {total} matching PR(s) across {len(all_results)} repo(s).")


if __name__ == "__main__":
    main()