#!/usr/bin/env python3
"""
Mine GitHub for agentic-AI Context Files and Skill files whose Markdown
headings are dedicated to code duplication / code reuse topics.

Scope of files collected:
  Context Files:
    - AGENTS.md
    - CLAUDE.md
    - GEMINI.md
    - .cursorrules
    - .github/copilot-instructions.md

  Skill files:
    - SKILL.md
    - SKILLS.md  # kept for tolerance, although SKILL.md is the common convention

Strategy:
  Use GitHub Code Search to find candidate files whose content mentions
  code-duplication terms, keep only the target Context File / Skill paths,
  filter repositories by software-quality criteria, fetch each file's full
  content, and keep only files where a CLONE_TERM appears in a Markdown heading.

Quality criteria (all must pass):
  - Not a fork
  - Has a license
  - >= 100 stars  (configurable via --min-stars)
  - Created >= 2 months ago
  - Last push within the past 6 months
  - >= 2 contributors
  - >= 1 pull request (any state)

Classification:
  - "dedicated" : a CLONE_TERM appears in the main heading (# ...)
  - "partial"   : a CLONE_TERM appears in a subheading (## ... or deeper)
  - "mention"   : a CLONE_TERM appears only in body text / bullet points

  Files where no CLONE_TERM is found anywhere are discarded.

Output:
  context_skill_heading_results/
    context_skill_duplication_report.json    — per-file detail
    context_skill_duplication_analysis.json  — summary / statistics

Usage:
    export GITHUB_TOKEN="ghp_..."
    python search_context_files_and_skills_duplication.py
    python search_context_files_and_skills_duplication.py --max-search-pages 5 --min-stars 200
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Token pool – rotate across multiple GITHUB_TOKEN_N env vars on rate-limit
# ---------------------------------------------------------------------------

class TokenPool:
    """Round-robin pool of GitHub personal access tokens."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = [t for t in tokens if t]
        self._index = 0

    def __len__(self) -> int:
        return len(self._tokens)

    @property
    def current(self) -> str | None:
        return self._tokens[self._index] if self._tokens else None

    def rotate(self) -> str | None:
        """Advance to the next token (wraps around). Returns the new token."""
        if not self._tokens:
            return None
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"

RESULTS_DIR   = Path("2_context_github_results")
REPORT_PATH   = RESULTS_DIR / "clone_terms_report.json"
ANALYSIS_PATH = RESULTS_DIR / "clone_terms_analysis.json"
CHECKPOINT_PATH = RESULTS_DIR / "clone_terms_checkpoint.json"

# Software-quality thresholds
MIN_STARS         = 100
MIN_AGE_MONTHS    = 2
MAX_STALE_MONTHS  = 6    # last push must be within this many months
MIN_CONTRIBUTORS  = 2
MIN_PULL_REQUESTS = 1

SLEEP_BETWEEN_REQUESTS = 1.0   # seconds between successful API calls
MAX_SEARCH_PAGES       = 10    # pages per query (100 results/page)
SNIPPET_CONTEXT        = 80    # characters around each heading match

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

CLEAN_TERMS: list[str] = CLONE_TERMS

# Target Context Files and Skill files.
# - File-name targets can appear anywhere in the repository tree.
# - Strict path targets must match the normalized repo-relative path exactly.
CONTEXT_FILE_NAMES: set[str] = {
    "AGENTS.md",
    "CLAUDE.md",
    "GEMINI.md",
    ".cursorrules",
}

CONTEXT_FILE_PATHS: set[str] = {
    ".github/copilot-instructions.md",
}

SKILL_FILE_NAMES: set[str] = {
    "SKILL.md",
    "SKILLS.md",  # tolerated spelling variant
}

# GitHub Code Search does not use our local heading regex, so we first search
# by term + target filename/path, then fetch the file and apply the heading regex.
CODE_SEARCH_TARGET_QUALIFIERS: list[str] = [
    "filename:AGENTS.md",
    "filename:CLAUDE.md",
    "filename:GEMINI.md",
    "filename:.cursorrules",
    "filename:copilot-instructions.md path:.github",
    "filename:SKILL.md",
    "filename:SKILLS.md",
]

# Heading-level prefixes sent to GitHub Code Search.
# "# term"  → only h1 (dedicated); "## term" → only h2; etc.
# Each prefix generates independent queries so GitHub pre-filters to files
# where the term actually appears inside a Markdown heading.
_HEADING_PREFIXES: list[str] = ["# ", "## ", "### "]

CODE_SEARCH_QUERIES: list[str] = [
    f'"{prefix}{term}" in:file {target}'
    for target in CODE_SEARCH_TARGET_QUALIFIERS
    for term in CLONE_TERMS
    for prefix in _HEADING_PREFIXES
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FileCandidate(NamedTuple):
    repo_full_name: str
    file_path:      str
    html_url:       str
    found_by_query: str
    artifact_type:  str  # "context_file" | "skill_file"


class TermHit(NamedTuple):
    term:     str
    count:    int
    snippets: list[str]


class FileResult(NamedTuple):
    candidate:      FileCandidate
    repo_stars:     int
    repo_created_at: str
    repo_pushed_at: str
    repo_license:   str
    repo_language:  str
    hits:           list[TermHit]
    total_matches:  int
    classification: str   # "dedicated" | "partial" | "mention"
    raw_url:        str


def candidate_key(repo_full_name: str, file_path: str) -> str:
    return f"{repo_full_name}::{file_path}"


def serialize_file_result(result: FileResult) -> dict[str, Any]:
    return {
        "candidate": {
            "repo_full_name": result.candidate.repo_full_name,
            "file_path": result.candidate.file_path,
            "html_url": result.candidate.html_url,
            "found_by_query": result.candidate.found_by_query,
            "artifact_type": result.candidate.artifact_type,
        },
        "repo_stars": result.repo_stars,
        "repo_created_at": result.repo_created_at,
        "repo_pushed_at": result.repo_pushed_at,
        "repo_license": result.repo_license,
        "repo_language": result.repo_language,
        "hits": [
            {
                "term": hit.term,
                "count": hit.count,
                "snippets": hit.snippets,
            }
            for hit in result.hits
        ],
        "total_matches": result.total_matches,
        "classification": result.classification,
        "raw_url": result.raw_url,
    }


def deserialize_file_result(data: dict[str, Any]) -> FileResult:
    candidate = data["candidate"]
    return FileResult(
        candidate=FileCandidate(
            repo_full_name=candidate["repo_full_name"],
            file_path=candidate["file_path"],
            html_url=candidate["html_url"],
            found_by_query=candidate["found_by_query"],
            artifact_type=candidate["artifact_type"],
        ),
        repo_stars=data.get("repo_stars", 0),
        repo_created_at=data.get("repo_created_at", ""),
        repo_pushed_at=data.get("repo_pushed_at", ""),
        repo_license=data.get("repo_license", ""),
        repo_language=data.get("repo_language", ""),
        hits=[
            TermHit(
                term=hit["term"],
                count=hit["count"],
                snippets=hit.get("snippets", []),
            )
            for hit in data.get("hits", [])
        ],
        total_matches=data.get("total_matches", 0),
        classification=data.get("classification", "mention"),
        raw_url=data.get("raw_url", ""),
    )


def load_checkpoint() -> dict[str, Any]:
    if not CHECKPOINT_PATH.exists():
        return {
            "candidates": None,
            "repo_quality": {},
            "file_outcomes": {},
        }

    with CHECKPOINT_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)

    return {
        "candidates": data.get("candidates"),
        "repo_quality": data.get("repo_quality", {}),
        "file_outcomes": data.get("file_outcomes", {}),
    }


def save_checkpoint(data: dict[str, Any]) -> None:
    tmp_path = CHECKPOINT_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(CHECKPOINT_PATH)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_stats(results: list[FileResult]) -> dict:
    """Return term-file-count mapping, classification counts, and total matched files."""
    term_file_counts: dict[str, int] = {t: 0 for t in CLEAN_TERMS}
    classification_counts: dict[str, int] = {"dedicated": 0, "partial": 0, "mention": 0}
    for result in results:
        for hit in result.hits:
            term_file_counts[hit.term] += 1
        classification_counts[result.classification] = (
            classification_counts.get(result.classification, 0) + 1
        )
    return {
        "total_md_files_with_match": len(results),
        "classification_counts": classification_counts,
        "term_file_counts": term_file_counts,
        "top_terms": sorted(
            [{"term": t, "files": c} for t, c in term_file_counts.items() if c > 0],
            key=lambda x: x["files"],
            reverse=True,
        ),
    }


# ---------------------------------------------------------------------------
# Target-file filtering
# ---------------------------------------------------------------------------

def normalize_repo_path(file_path: str) -> str:
    return file_path.replace("\\", "/").lstrip("/")


def classify_artifact_path(file_path: str) -> str | None:
    """Return artifact type for target files; return None for everything else."""
    normalized = normalize_repo_path(file_path)
    filename = normalized.rsplit("/", 1)[-1]

    if normalized in CONTEXT_FILE_PATHS:
        return "context_file"

    if filename in CONTEXT_FILE_NAMES:
        return "context_file"

    if filename in SKILL_FILE_NAMES:
        return "skill_file"

    return None


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Accept":               "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent":           "context-skill-duplication-research-miner",
    })
    token = _token_pool.current
    if token:
        session.headers["Authorization"] = f"Bearer {token}"
        print(f"  Token pool: {len(_token_pool)} token(s) loaded.", file=sys.stderr)
    else:
        print(
            "[WARNING] No GITHUB_TOKEN found – unauthenticated requests are "
            "severely rate-limited (10 code-search req/min).",
            file=sys.stderr,
        )
    return session


def request_json(
    session: requests.Session,
    url: str,
    params: dict | None = None,
    max_retries: int = 5,
) -> Any:
    for attempt in range(max_retries):
        try:
            response = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"  [REQUEST ERROR] {exc}", file=sys.stderr)
            return None

        if response.status_code in (403, 429):
            # Try rotating to the next token before sleeping
            if len(_token_pool) > 1:
                new_token = _token_pool.rotate()
                session.headers["Authorization"] = f"Bearer {new_token}"
                print(
                    f"  [RATE LIMIT] status={response.status_code}; "
                    f"rotating to next token (attempt {attempt + 1}/{max_retries}) ...",
                    file=sys.stderr,
                )
                continue

            # Only one token — fall back to sleeping
            retry_after = response.headers.get("Retry-After")
            remaining   = response.headers.get("x-ratelimit-remaining")
            reset       = response.headers.get("x-ratelimit-reset")

            if retry_after:
                wait = int(retry_after) + 2
            elif remaining == "0" and reset:
                wait = max(0, int(reset) - int(time.time())) + 2
            else:
                wait = min(60 * (attempt + 1), 300)

            print(
                f"  [RATE LIMIT] status={response.status_code}; "
                f"waiting {wait}s before retrying ...",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

        if response.status_code == 404:
            return None

        if response.status_code >= 400:
            print(
                f"  [API ERROR] {response.status_code}: {response.text[:200]}",
                file=sys.stderr,
            )
            return None

        time.sleep(SLEEP_BETWEEN_REQUESTS)
        return response.json()

    return None


# ---------------------------------------------------------------------------
# Step 1 – Code search
# ---------------------------------------------------------------------------

def search_code_files(
    session: requests.Session,
    queries: list[str],
    max_pages: int,
) -> list[FileCandidate]:
    """Run all CODE_SEARCH_QUERIES and collect unique target (repo, file) candidates."""
    seen:       set[tuple[str, str]] = set()
    candidates: list[FileCandidate]  = []

    for query in queries:
        print(f"\n[SEARCH] {query}", file=sys.stderr)

        for page in range(1, max_pages + 1):
            data = request_json(
                session,
                f"{GITHUB_API}/search/code",
                params={"q": query, "per_page": 100, "page": page},
            )

            if not data:
                break

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                repo      = item.get("repository", {}).get("full_name", "")
                file_path = item.get("path", "")
                html_url  = item.get("html_url", "")

                if not repo or not file_path:
                    continue

                artifact_type = classify_artifact_path(file_path)
                if artifact_type is None:
                    continue

                key = (repo, file_path)
                if key not in seen:
                    seen.add(key)
                    candidates.append(FileCandidate(
                        repo_full_name=repo,
                        file_path=file_path,
                        html_url=html_url,
                        found_by_query=query,
                        artifact_type=artifact_type,
                    ))

            print(
                f"  page {page}: {len(items)} items  "
                f"(unique target candidates so far: {len(candidates):,})",
                file=sys.stderr,
            )

            if len(items) < 100:
                break

    return candidates


# ---------------------------------------------------------------------------
# Step 2 – Repository quality check
# ---------------------------------------------------------------------------

_repo_cache: dict[str, dict | None] = {}


def check_repo_quality(
    session: requests.Session,
    full_name: str,
    min_stars: int,
) -> dict | None:
    """
    Return a repo metadata dict if the repo satisfies all quality criteria.
    Results are cached to avoid repeated API calls for the same repository.
    """
    if full_name in _repo_cache:
        return _repo_cache[full_name]

    data = request_json(session, f"{GITHUB_API}/repos/{full_name}")
    if not data:
        _repo_cache[full_name] = None
        return None

    now = datetime.now(timezone.utc)

    # ── Checks that require only the repo endpoint ──────────────────────────
    if data.get("fork"):
        _repo_cache[full_name] = None
        return None

    if not data.get("license"):
        _repo_cache[full_name] = None
        return None

    if data.get("stargazers_count", 0) < min_stars:
        _repo_cache[full_name] = None
        return None

    try:
        created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        pushed_at  = datetime.fromisoformat(data["pushed_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError):
        _repo_cache[full_name] = None
        return None

    age_months   = (now - created_at).days / 30
    stale_months = (now - pushed_at).days / 30

    if age_months < MIN_AGE_MONTHS:
        _repo_cache[full_name] = None
        return None

    if stale_months > MAX_STALE_MONTHS:
        _repo_cache[full_name] = None
        return None

    # ── Contributors check (extra API call) ─────────────────────────────────
    contributors = request_json(
        session,
        f"{GITHUB_API}/repos/{full_name}/contributors",
        params={"per_page": MIN_CONTRIBUTORS, "anon": "1"},
    )
    if not contributors or len(contributors) < MIN_CONTRIBUTORS:
        _repo_cache[full_name] = None
        return None

    # ── Pull requests check (extra API call) ────────────────────────────────
    prs = request_json(
        session,
        f"{GITHUB_API}/repos/{full_name}/pulls",
        params={"state": "all", "per_page": 1},
    )
    if not prs:
        _repo_cache[full_name] = None
        return None

    result = {
        "full_name":   full_name,
        "stars":       data.get("stargazers_count", 0),
        "created_at":  data.get("created_at", ""),
        "pushed_at":   data.get("pushed_at", ""),
        "license":     (data.get("license") or {}).get("spdx_id") or (data.get("license") or {}).get("name", ""),
        "language":    data.get("language") or "",
        "description": data.get("description") or "",
    }
    _repo_cache[full_name] = result
    return result


# ---------------------------------------------------------------------------
# Step 3 – Fetch file content
# ---------------------------------------------------------------------------

def fetch_file_content(
    session: requests.Session,
    full_name: str,
    file_path: str,
) -> tuple[str | None, str]:
    """
    Fetch raw text of a file via the GitHub Contents API.
    Returns (content_text, raw_url). raw_url is empty on failure.
    """
    data = request_json(
        session,
        f"{GITHUB_API}/repos/{full_name}/contents/{file_path}",
    )
    if not data or data.get("type") != "file":
        return None, ""

    raw_url = data.get("download_url", "")

    encoded = data.get("content", "")
    if not encoded:
        return None, raw_url

    try:
        content = base64.b64decode(encoded).decode("utf-8", errors="replace")
    except Exception:
        return None, raw_url

    return content, raw_url


# ---------------------------------------------------------------------------
# Step 4 – Heading regex + snippets
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def extract_heading_snippet(text: str, start: int, end: int, context: int = SNIPPET_CONTEXT) -> str:
    snippet_start = max(0, start - context)
    snippet_end   = min(len(text), end + context)
    return "..." + text[snippet_start:snippet_end].replace("\n", " ").strip() + "..."


def _extract_body_hits(text: str) -> list[TermHit]:
    """Find CLONE_TERMS anywhere in the file text (used for 'mention' classification)."""
    hits: list[TermHit] = []
    for term in CLEAN_TERMS:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        snippets: list[str] = []
        seen: set[str] = set()
        for m in matches:
            start = max(0, m.start() - SNIPPET_CONTEXT)
            end = min(len(text), m.end() + SNIPPET_CONTEXT)
            snippet = "..." + text[start:end].replace("\n", " ").strip() + "..."
            if snippet not in seen:
                seen.add(snippet)
                snippets.append(snippet)
        hits.append(TermHit(term=term, count=len(matches), snippets=snippets))
    return hits


def classify_and_extract_heading_hits(text: str) -> tuple[str | None, list[TermHit]]:
    """
    Classification rule:
      - "dedicated" : a CLONE_TERM appears in the main heading (# ...)
      - "partial"   : a CLONE_TERM appears in a subheading (## ... or deeper)
      - "mention"   : a CLONE_TERM appears only in body text / bullet points
      - None        : no CLONE_TERM found anywhere → file is discarded
    """
    term_counts: dict[str, int] = {term: 0 for term in CLEAN_TERMS}
    term_snippets: dict[str, list[str]] = {term: [] for term in CLEAN_TERMS}
    term_seen_snippets: dict[str, set[str]] = {term: set() for term in CLEAN_TERMS}

    has_dedicated = False
    has_partial = False

    for heading_match in _HEADING_RE.finditer(text):
        heading_line = heading_match.group(0)
        heading_level = len(heading_match.group(1))

        for term in CLEAN_TERMS:
            if not re.search(re.escape(term), heading_line, re.IGNORECASE):
                continue

            term_counts[term] += 1
            snippet = extract_heading_snippet(text, heading_match.start(), heading_match.end())
            if snippet not in term_seen_snippets[term]:
                term_seen_snippets[term].add(snippet)
                term_snippets[term].append(snippet)

            if heading_level == 1:
                has_dedicated = True
            else:
                has_partial = True

    hits = [
        TermHit(term=term, count=count, snippets=term_snippets[term])
        for term, count in term_counts.items()
        if count > 0
    ]

    if has_dedicated:
        return "dedicated", hits

    if has_partial:
        return "partial", hits

    # No heading match — check body text for a 'mention'
    body_hits = _extract_body_hits(text)
    if body_hits:
        return "mention", body_hits

    return None, []


# ---------------------------------------------------------------------------
# Step 5 – Save report (detailed, one entry per file)
# ---------------------------------------------------------------------------

def save_report(
    results: list[FileResult],
    report_path: Path,
    total_records: int,
    total_md_records: int,
) -> None:
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "csv": "github_code_search",
                "total_records": total_records,
                "total_md_records": total_md_records,
                "total_md_files_with_match": len(results),
                "matches": [
                    {
                        "repo_name": r.candidate.repo_full_name,
                        "file_name": Path(r.candidate.file_path).name,
                        "file_path": r.candidate.file_path,
                        "github_link": r.candidate.html_url,
                        "raw_url": r.raw_url,
                        "timestamp": r.repo_pushed_at,
                        "branch": "",
                        "created_at": r.repo_created_at,
                        "#commits": "",
                        "is_empty": "FALSE",
                        "first_commit_sha": "",
                        "last_commit_sha": "",
                        "total_occurrences": r.total_matches,
                        "classification": r.classification,
                        "artifact_type": r.candidate.artifact_type,
                        "found_by_query": r.candidate.found_by_query,
                        "repo_stars": r.repo_stars,
                        "repo_license": r.repo_license,
                        "repo_language": r.repo_language,
                        "terms": [
                            {
                                "term": hit.term,
                                "count": hit.count,
                                "snippets": hit.snippets[:3],
                            }
                            for hit in r.hits
                        ],
                    }
                    for r in results
                ],
            }
        ],
    }
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] Report   → {report_path}")


# ---------------------------------------------------------------------------
# Step 6 – Save analysis (summary / statistics)
# ---------------------------------------------------------------------------

def save_analysis(
    results: list[FileResult],
    analysis_path: Path,
    total_records: int,
    total_md_records: int,
) -> None:
    stats = aggregate_stats(results)

    artifact_type_counts: dict[str, int] = {"context_file": 0, "skill_file": 0}
    term_counts:  dict[str, int] = {t: 0 for t in CLEAN_TERMS}
    repo_counts:  dict[str, int] = {}
    lang_counts:  dict[str, int] = {}

    for r in results:
        artifact_type_counts[r.candidate.artifact_type] = (
            artifact_type_counts.get(r.candidate.artifact_type, 0) + 1
        )
        for hit in r.hits:
            term_counts[hit.term] += 1
        repo_counts[r.candidate.repo_full_name] = (
            repo_counts.get(r.candidate.repo_full_name, 0) + 1
        )
        lang = r.repo_language or "unknown"
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "global_summary": {
            "total_records_across_all_csvs": total_records,
            "total_md_files_across_all_csvs": total_md_records,
            "total_md_files_with_match": stats["total_md_files_with_match"],
            "classification_counts": stats["classification_counts"],
            "top_terms_globally": stats["top_terms"],
        },
        "per_csv": [
            {
                "csv": "github_code_search",
                "total_records_in_csv": total_records,
                "total_md_files": total_md_records,
                "md_files_with_match": stats["total_md_files_with_match"],
                "classification_counts": stats["classification_counts"],
                "top_terms": stats["top_terms"],
                "artifact_type_counts": artifact_type_counts,
                "top_repos": sorted(
                    [{"repo": r, "files": c} for r, c in repo_counts.items()],
                    key=lambda x: x["files"],
                    reverse=True,
                )[:20],
                "by_language": sorted(
                    [{"language": language, "files": count} for language, count in lang_counts.items()],
                    key=lambda x: x["files"],
                    reverse=True,
                ),
            }
        ],
    }
    analysis_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] Analysis → {analysis_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = perf_counter()

    session = build_session()
    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint = load_checkpoint()

    # ── Step 1 – Collect candidate (repo, file) pairs via code search ────────
    print("=" * 60)
    print("  Step 1: GitHub Code Search")
    print("=" * 60)
    print(f"  Search queries: {len(CODE_SEARCH_QUERIES):,}")
    if checkpoint["candidates"]:
        candidates = [
            FileCandidate(**candidate)
            for candidate in checkpoint["candidates"]
        ]
        print(f"  Checkpoint loaded - reusing {len(candidates):,} previously discovered candidates.")
    else:
        candidates = search_code_files(session, CODE_SEARCH_QUERIES, MAX_SEARCH_PAGES)
        checkpoint["candidates"] = [candidate._asdict() for candidate in candidates]
        save_checkpoint(checkpoint)
        print(f"  Checkpoint saved   - candidate discovery persisted to {CHECKPOINT_PATH}")
    print(f"\n  Unique target (repo, file) candidates: {len(candidates):,}")

    artifact_counts: dict[str, int] = {}
    for c in candidates:
        artifact_counts[c.artifact_type] = artifact_counts.get(c.artifact_type, 0) + 1
    print(f"  Candidate artifact types: {artifact_counts}")

    # ── Step 2 – Filter repos by quality criteria ───────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2: Repository quality check")
    print("=" * 60)
    unique_repos = sorted({c.repo_full_name for c in candidates})
    print(f"  Unique repositories to evaluate: {len(unique_repos):,}")
    print(f"  Quality thresholds: stars>={MIN_STARS}, age>={MIN_AGE_MONTHS}mo, "
          f"active<={MAX_STALE_MONTHS}mo, contributors>={MIN_CONTRIBUTORS}, PRs>={MIN_PULL_REQUESTS}\n")

    quality_repos: set[str] = set()
    repo_quality_checkpoint: dict[str, Any] = checkpoint["repo_quality"]

    for repo_name, outcome in repo_quality_checkpoint.items():
        if outcome.get("passed"):
            quality_repos.add(repo_name)
            repo_info = outcome.get("repo_info") or {}
            if repo_info:
                _repo_cache[repo_name] = repo_info
        else:
            _repo_cache[repo_name] = None

    w = len(str(len(unique_repos)))

    for completed, repo in enumerate(unique_repos, 1):
        if repo in repo_quality_checkpoint:
            info = repo_quality_checkpoint[repo].get("repo_info") if repo_quality_checkpoint[repo].get("passed") else None
        else:
            info = check_repo_quality(session, repo, MIN_STARS)
            repo_quality_checkpoint[repo] = {
                "passed": info is not None,
                "repo_info": info,
            }
            save_checkpoint(checkpoint)
        status = "PASS" if info else "FAIL"
        print(f"  [{completed:>{w}}/{len(unique_repos)}] {status}  {repo}")
        if info:
            quality_repos.add(repo)

    print(f"\n  Repositories passing quality check: {len(quality_repos):,}")

    # ── Step 3-4 – Fetch content, search headings, classify ─────────────────
    print("\n" + "=" * 60)
    print("  Step 3: Fetch content → heading regex → classify")
    print("=" * 60)
    quality_candidates = [c for c in candidates if c.repo_full_name in quality_repos]
    print(f"  Files to process: {len(quality_candidates):,}\n")

    file_outcomes: dict[str, Any] = checkpoint["file_outcomes"]
    results: list[FileResult] = []
    for outcome in file_outcomes.values():
        if outcome.get("status") == "matched" and outcome.get("result"):
            results.append(deserialize_file_result(outcome["result"]))

    n = len(quality_candidates)
    w = len(str(n)) if n else 1

    for completed, candidate in enumerate(quality_candidates, 1):
        print(f"  [{completed:>{w}}/{n}] processing...", end="\r")
        key = candidate_key(candidate.repo_full_name, candidate.file_path)

        if key in file_outcomes:
            outcome = file_outcomes[key]
            status = outcome.get("status")
            if status == "matched":
                restored = deserialize_file_result(outcome["result"])
                print(
                    f"  {candidate.repo_full_name} / {candidate.file_path}  "
                    f"→ [{restored.classification}] [{candidate.artifact_type}] restored from checkpoint."
                )
            elif status == "no_match":
                print(
                    f"  {candidate.repo_full_name} / {candidate.file_path}  "
                    "→ No CLONE_TERM found anywhere. [checkpoint]"
                )
            else:
                print(
                    f"  {candidate.repo_full_name} / {candidate.file_path}  "
                    "→ Could not fetch content. [checkpoint]"
                )
            continue

        content, raw_url = fetch_file_content(
            session, candidate.repo_full_name, candidate.file_path
        )
        if content is None:
            file_outcomes[key] = {
                "status": "fetch_failed",
                "repo_full_name": candidate.repo_full_name,
                "file_path": candidate.file_path,
            }
            save_checkpoint(checkpoint)
            print(f"  {candidate.repo_full_name} / {candidate.file_path}  → Could not fetch content.")
            continue

        classification, hits = classify_and_extract_heading_hits(content)
        if classification is None or not hits:
            file_outcomes[key] = {
                "status": "no_match",
                "repo_full_name": candidate.repo_full_name,
                "file_path": candidate.file_path,
            }
            save_checkpoint(checkpoint)
            print(
                f"  {candidate.repo_full_name} / {candidate.file_path}  "
                "→ No CLONE_TERM found anywhere."
            )
            continue

        n_hits = sum(h.count for h in hits)
        repo_info = _repo_cache.get(candidate.repo_full_name) or {}

        print(
            f"  {candidate.repo_full_name} / {candidate.file_path}  "
            f"→ [{classification}] [{candidate.artifact_type}] "
            f"{len(hits)} term(s), {n_hits} heading occurrence(s)."
        )

        results.append(FileResult(
            candidate=candidate,
            repo_stars=repo_info.get("stars", 0),
            repo_created_at=repo_info.get("created_at", ""),
            repo_pushed_at=repo_info.get("pushed_at", ""),
            repo_license=repo_info.get("license", ""),
            repo_language=repo_info.get("language", ""),
            hits=hits,
            total_matches=n_hits,
            classification=classification,
            raw_url=raw_url,
        ))
        file_outcomes[key] = {
            "status": "matched",
            "result": serialize_file_result(results[-1]),
        }
        save_checkpoint(checkpoint)

    print()  # newline after the \r progress line

    # ── Summary ──────────────────────────────────────────────────────────────
    counts = {"dedicated": 0, "partial": 0, "mention": 0}
    artifact_type_counts = {"context_file": 0, "skill_file": 0}
    for r in results:
        counts[r.classification] += 1
        artifact_type_counts[r.candidate.artifact_type] += 1

    print(f"\n{'='*60}")
    print(f"  Files matched: {len(results):,}")
    print(f"    dedicated:              {counts['dedicated']:,}")
    print(f"    partial:                {counts['partial']:,}")
    print(f"    mention:                {counts['mention']:,}")
    print(f"    context_file:           {artifact_type_counts['context_file']:,}")
    print(f"    skill_file:             {artifact_type_counts['skill_file']:,}")

    # ── Save output ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Saving output files ...")
    print(f"{'='*60}")
    save_report(
        results,
        REPORT_PATH,
        total_records=len(candidates),
        total_md_records=len(quality_candidates),
    )
    save_analysis(
        results,
        ANALYSIS_PATH,
        total_records=len(candidates),
        total_md_records=len(quality_candidates),
    )

    elapsed = perf_counter() - start
    m, s = divmod(elapsed, 60)
    print(f"\n{'='*60}")
    if m:
        print(f"  Total runtime: {int(m)}m {s:.1f}s")
    else:
        print(f"  Total runtime: {s:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
