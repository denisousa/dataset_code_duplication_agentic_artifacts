"""
Analyze all AI_CONFIG CSV files (commands, context_files, mcp, skills, subagents)
to identify .md files that contain code-clone / code-duplication related terms.

Workflow:
  1. For each configured CSV source, load the file and filter rows whose file
     path ends with .md.
  2. Convert each github_link to a raw-content URL and fetch the file text.
  3. Apply a case-insensitive regex for every CLONE_TERM.
  4. Collect per-file statistics (matched terms, counts, snippets).
  5. Write detailed results to clone_terms_report.json.
  6. Write a summary analysis to clone_terms_analysis.json.

Requires GITHUB_TOKEN in environment (or .env file) to avoid rate-limiting.
"""

import csv
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AI_CONFIG_DIR = Path("ai_config")
REPOS_CSV = Path("ai_config/repos.csv")
LOCAL_REPOS_DIR = AI_CONFIG_DIR / "repos_data" / "repos"

# Maps each CSV filename to the column that holds the file path.
CSV_SOURCES: dict[str, str] = {
    "commands.csv":      "command",
    "context_files.csv": "context_file",
    "hooks.csv":         "hook",
    "mcp.csv":           "mcp",
    "rules.csv":         "rule",
    "settings.csv":      "settings",
    "skills.csv":        "skills.md",
    "subagents.csv":     "subagent",
}

RESULTS_DIR   = Path("1_configurations_artifacts_results")
REPORT_PATH   = RESULTS_DIR / "clone_terms_report.json"
ANALYSIS_PATH = RESULTS_DIR / "clone_terms_analysis.json"
FAILED_PATH   = RESULTS_DIR / "clone_terms_failed.json"

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

# Adicionar resue and code reused

CLEAN_TERMS: list[str] = CLONE_TERMS

MAX_WORKERS = 20            # concurrent HTTP fetch threads
SNIPPET_CONTEXT = 100       # characters of context before and after each match
DELAY_BETWEEN_REQUESTS = 0  # no artificial delay; parallelism + auto rate-limit handles pacing


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FileRecord(NamedTuple):
    repo_name: str
    file_path: str
    github_link: str
    name: str
    language: str
    commit_sha: str
    timestamp: str
    branch: str
    created_at: str
    num_commits: str
    is_empty: str
    first_commit_sha: str


class TermHit(NamedTuple):
    term: str
    count: int
    snippets: list[str]


class MatchResult(NamedTuple):
    record: FileRecord
    hits: list[TermHit]
    total_matches: int
    raw_url: str


class FailedRecord(NamedTuple):
    csv: str
    repo_name: str
    file_path: str
    raw_url: str
    reason: str  # "no_github_link" | "fetch_error"


# ---------------------------------------------------------------------------
# Step 0 – Load engineered repos
# ---------------------------------------------------------------------------

def load_engineered_repos(repos_path: Path) -> set[str]:
    """Return the set of repo_names where engineered_project is True."""
    engineered: set[str] = set()
    rows = load_csv(repos_path)
    for row in rows:
        if row.get("engineered_project", "").strip().lower() == "true":
            engineered.add(row.get("repo_name", "").strip())
    return engineered


# ---------------------------------------------------------------------------
# Step 1 – Load a CSV file
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_csv(csv_path: Path) -> list[dict]:
    """Read all rows from a CSV file and return them as a list of dicts."""
    csv.field_size_limit(10 * 1024 * 1024)  # 10 MB — handles large commit messages
    rows: list[dict] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Step 2 – Filter .md records
# ---------------------------------------------------------------------------

def filter_records(
    rows: list[dict],
    file_col: str,
    engineered_repos: set[str],
) -> list[FileRecord]:
    """Keep rows whose file-path column is non-empty and whose repo is an engineered project."""
    records: list[FileRecord] = []
    for row in rows:
        file_path: str = row.get(file_col, "").strip()
        repo_name: str = row.get("repo_name", "").strip()
        if file_path and repo_name in engineered_repos:
            records.append(
                FileRecord(
                    repo_name=row.get("repo_name", "").strip(),
                    file_path=file_path,
                    github_link=row.get("github_link", "").strip(),
                    name=row.get("name", "").strip(),
                    language=row.get("language", "").strip(),
                    commit_sha=row.get("last_commit_sha", "").strip(),
                    timestamp=row.get("timestamp", "").strip(),
                    branch=row.get("branch", "").strip(),
                    created_at=row.get("created_at", "").strip(),
                    num_commits=row.get("#commits", "").strip(),
                    is_empty=row.get("is_empty", "").strip(),
                    first_commit_sha=row.get("first_commit_sha", "").strip(),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Step 3 – Convert github_link to raw content URL
# ---------------------------------------------------------------------------

def github_link_to_raw_url(github_link: str, commit_sha: str = "") -> str | None:
    """
    Convert a GitHub tree URL to a raw.githubusercontent.com URL.

    When commit_sha is provided the branch segment is replaced with the SHA,
    producing a stable, immutable URL:
      https://github.com/owner/repo/tree/branch/path/to/file.md
      → https://raw.githubusercontent.com/owner/repo/{sha}/path/to/file.md

    Without a SHA the branch name is kept as-is (fallback).
    """
    if not github_link:
        return None
    raw = github_link.replace("https://github.com/", "https://raw.githubusercontent.com/")
    if commit_sha:
        raw = re.sub(r"/tree/[^/]+/", f"/{commit_sha}/", raw, count=1)
    else:
        raw = re.sub(r"/tree/", "/", raw, count=1)
    return raw


# ---------------------------------------------------------------------------
# Step 4 – Fetch file content
# ---------------------------------------------------------------------------

def build_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    headers: dict = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        print(
            "[WARNING] GITHUB_TOKEN not set. "
            "Unauthenticated requests are limited to 60/hour.\n"
        )
    return headers


_rate_limit_lock = threading.Lock()


def handle_rate_limit(response: requests.Response) -> None:
    reset_ts = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
    with _rate_limit_lock:
        wait = max(reset_ts - time.time(), 0) + 5
        if wait > 0:
            print(f"  [RATE LIMIT] Waiting {wait:.0f}s before retrying...")
            time.sleep(wait)


def fetch_raw_content(raw_url: str, headers: dict) -> str | None:
    """Fetch plain text from a raw.githubusercontent.com URL with retry on rate-limit."""
    while True:
        try:
            response = requests.get(raw_url, headers=headers, timeout=15)
        except requests.RequestException as exc:
            print(f"  [REQUEST ERROR] {exc}")
            return None

        if response.status_code == 200:
            return response.text

        if response.status_code in (429, 403) and "rate limit" in response.text.lower():
            handle_rate_limit(response)
            continue

        return None  # 404 or other non-retryable error


def build_local_repo_path(repo_name: str) -> Path:
    return LOCAL_REPOS_DIR / repo_name.replace("/", "§")


def read_text_with_fallbacks(path: Path) -> str:
    raw_bytes = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def read_local_content(record: FileRecord) -> str | None:
    local_path = build_local_repo_path(record.repo_name) / record.file_path
    try:
        return read_text_with_fallbacks(local_path)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Step 5 – Apply regex and extract statistics
# ---------------------------------------------------------------------------

def extract_snippets(text: str, term: str, context: int = SNIPPET_CONTEXT) -> list[str]:
    """Return a deduplicated list of text snippets around each match of *term*."""
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    snippets: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        start = max(0, match.start() - context)
        end = min(len(text), match.end() + context)
        snippet = "..." + text[start:end].replace("\n", " ").strip() + "..."
        if snippet not in seen:
            seen.add(snippet)
            snippets.append(snippet)
    return snippets


def apply_regex(text: str) -> list[TermHit]:
    """Return TermHit objects for every CLONE_TERM found in *text*."""
    hits: list[TermHit] = []
    for term in CLEAN_TERMS:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        matches = pattern.findall(text)
        if matches:
            snippets = extract_snippets(text, term)
            hits.append(TermHit(term=term, count=len(matches), snippets=snippets))
    return hits


# ---------------------------------------------------------------------------
# Step 6 – Process a single CSV source
# ---------------------------------------------------------------------------

def _fetch_one(
    record: FileRecord,
    headers: dict,
) -> tuple[FileRecord, str, str | None]:
    """Read one record from the local mirror, falling back to raw_url when needed."""
    raw_url = github_link_to_raw_url(record.github_link, record.commit_sha)
    content = read_local_content(record)
    if content is None and raw_url:
        content = fetch_raw_content(raw_url, headers)
    return record, raw_url, content


def process_csv(csv_name: str, file_col: str, headers: dict, engineered_repos: set[str]) -> dict:
    """
    Load, filter, fetch, and search one CSV file.
    Returns a dict with all data needed for both output files.
    """
    csv_path = AI_CONFIG_DIR / csv_name
    print(f"\n{'='*60}")
    print(f"  CSV: {csv_name}  (file column: '{file_col}')")
    print(f"{'='*60}")

    rows = load_csv(csv_path)
    print(f"  Total records: {len(rows)}")

    records = filter_records(rows, file_col, engineered_repos)
    print(f"  Processable records: {len(records)}")

    results: list[MatchResult] = []
    failed: list[FailedRecord] = []
    n_records = len(records)
    completed = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_one, record, headers): record for record in records}
        for future in as_completed(futures):
            completed += 1
            record, raw_url, content = future.result()

            if not raw_url:
                print(f"  [{completed}/{n_records}] SKIP (no github_link): {record.file_path}")
                failed.append(FailedRecord(
                    csv=csv_name,
                    repo_name=record.repo_name,
                    file_path=record.file_path,
                    raw_url="",
                    reason="no_github_link",
                ))
                continue

            print(f"  [{completed}/{n_records}] {record.repo_name} / {record.file_path}")

            if content is None:
                print("    → Could not fetch content.")
                failed.append(FailedRecord(
                    csv=csv_name,
                    repo_name=record.repo_name,
                    file_path=record.file_path,
                    raw_url=raw_url,
                    reason="fetch_error",
                ))
                continue

            hits = apply_regex(content)
            if hits:
                n_hits = sum(h.count for h in hits)
                results.append(MatchResult(record=record, hits=hits, total_matches=n_hits, raw_url=raw_url))
                print(f"    → {len(hits)} term(s) matched, {n_hits} total occurrence(s).")

    return {
        "csv": csv_name,
        "total_records": len(rows),
        "total_md_records": len(records),
        "results": results,
        "failed": failed,
    }


# ---------------------------------------------------------------------------
# Step 7 – Aggregate per-CSV statistics
# ---------------------------------------------------------------------------

def aggregate_stats(results: list[MatchResult]) -> dict:
    """Return term-file-count mapping and total matched files."""
    term_file_counts: dict[str, int] = {t: 0 for t in CLEAN_TERMS}
    for result in results:
        for hit in result.hits:
            term_file_counts[hit.term] += 1
    return {
        "total_md_files_with_match": len(results),
        "term_file_counts": term_file_counts,
        "top_terms": sorted(
            [{"term": t, "files": c} for t, c in term_file_counts.items() if c > 0],
            key=lambda x: x["files"],
            reverse=True,
        ),
    }


# ---------------------------------------------------------------------------
# Step 8 – Save clone_terms_report.json  (detailed)
# ---------------------------------------------------------------------------

def save_report(csv_data: list[dict], report_path: Path) -> None:
    """Write one entry per CSV with all matched file details."""
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": [
            {
                "csv": entry["csv"],
                "total_records": entry["total_records"],
                "total_md_records": entry["total_md_records"],
                "total_md_files_with_match": len(entry["results"]),
                "matches": [
                    {
                        "repo_name": r.record.repo_name,
                        "file_name": Path(r.record.file_path).name,
                        "file_path": r.record.file_path,
                        "github_link": r.record.github_link,
                        "raw_url": r.raw_url,
                        "timestamp": r.record.timestamp,
                        "branch": r.record.branch,
                        "created_at": r.record.created_at,
                        "#commits": r.record.num_commits,
                        "is_empty": r.record.is_empty,
                        "first_commit_sha": r.record.first_commit_sha,
                        "last_commit_sha": r.record.commit_sha,
                        "total_occurrences": r.total_matches,
                        "terms": [
                            {
                                "term": hit.term,
                                "count": hit.count,
                                "snippets": hit.snippets[:3],
                            }
                            for hit in r.hits
                        ],
                    }
                    for r in entry["results"]
                ],
            }
            for entry in csv_data
        ],
    }

    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SAVED] Detailed report → {report_path}")


# ---------------------------------------------------------------------------
# Step 8b – Save clone_terms_failed.json
# ---------------------------------------------------------------------------

def save_failed_report(csv_data: list[dict], failed_path: Path) -> None:
    """Write all entries that could not be fetched to a separate JSON file."""
    all_failed: list[dict] = []
    for entry in csv_data:
        for f in entry.get("failed", []):
            all_failed.append({
                "csv":       f.csv,
                "repo_name": f.repo_name,
                "file_path": f.file_path,
                "raw_url":   f.raw_url,
                "reason":    f.reason,
            })

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_failed": len(all_failed),
        "failed": all_failed,
    }
    failed_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] Failed URLs report  → {failed_path}  ({len(all_failed)} entries)")


# ---------------------------------------------------------------------------
# Step 9 – Save clone_terms_analysis.json  (summary analysis)
# ---------------------------------------------------------------------------

def save_analysis(csv_data: list[dict], analysis_path: Path) -> None:
    """
    Write a summary analysis JSON with:
    - per-CSV: total rows, .md files count, matched files count, top terms
    - global: totals and overall term ranking
    """
    global_term_counts: dict[str, int] = {t: 0 for t in CLEAN_TERMS}
    global_total_records = 0
    global_total_md = 0
    global_total_matched = 0

    per_csv = []
    for entry in csv_data:
        stats = aggregate_stats(entry["results"])
        global_total_records += entry["total_records"]
        global_total_md      += entry["total_md_records"]
        global_total_matched += stats["total_md_files_with_match"]

        for term, count in stats["term_file_counts"].items():
            global_term_counts[term] += count

        per_csv.append({
            "csv": entry["csv"],
            "total_records_in_csv": entry["total_records"],
            "total_md_files": entry["total_md_records"],
            "md_files_with_match": stats["total_md_files_with_match"],
            "top_terms": stats["top_terms"],
        })

    global_top_terms = sorted(
        [{"term": t, "files": c} for t, c in global_term_counts.items() if c > 0],
        key=lambda x: x["files"],
        reverse=True,
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "global_summary": {
            "total_records_across_all_csvs": global_total_records,
            "total_md_files_across_all_csvs": global_total_md,
            "total_md_files_with_match": global_total_matched,
            "top_terms_globally": global_top_terms,
        },
        "per_csv": per_csv,
    }

    analysis_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] Analysis summary → {analysis_path}")


# ---------------------------------------------------------------------------
# Startup report
# ---------------------------------------------------------------------------

def print_startup_report(engineered_repos: set[str]) -> int:
    """
    Print a pre-run summary: entry count per CSV (engineered repos only) and total,
    plus an estimated runtime at 1 s/entry.
    Returns the total number of entries to be processed.
    """
    col_width = max(len(name) for name in CSV_SOURCES) + 2
    rows_data: list[tuple[str, int]] = []

    for csv_name, file_col in CSV_SOURCES.items():
        csv_path = AI_CONFIG_DIR / csv_name
        rows = load_csv(csv_path)
        count = sum(
            1 for row in rows
            if row.get(file_col, "").strip()
            and row.get("repo_name", "").strip() in engineered_repos
        )
        rows_data.append((csv_name, count))

    total = sum(count for _, count in rows_data)
    minutes, seconds = divmod(total * DELAY_BETWEEN_REQUESTS, 60)

    print("=" * 60)
    print("  Pre-run summary")
    print("=" * 60)
    print(f"  {'CSV file':<{col_width}}  {'Entries':>14}")
    print(f"  {'-'*col_width}  {'-'*14}")
    for csv_name, count in rows_data:
        print(f"  {csv_name:<{col_width}}  {count:>14,}")
    print(f"  {'-'*col_width}  {'-'*14}")
    print(f"  {'TOTAL':<{col_width}}  {total:>14,}")
    print()
    if minutes:
        print(f"  Estimated runtime: ~{int(minutes)}m {int(seconds)}s  ({total} entries × {DELAY_BETWEEN_REQUESTS}s)")
    else:
        print(f"  Estimated runtime: ~{int(seconds)}s  ({total} entries × {DELAY_BETWEEN_REQUESTS}s)")
    print("=" * 60)
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = perf_counter()

    engineered_repos = load_engineered_repos(REPOS_CSV)
    print(f"  Engineered repos loaded: {len(engineered_repos):,} from {REPOS_CSV}")

    print_startup_report(engineered_repos)

    headers = build_headers()

    RESULTS_DIR.mkdir(exist_ok=True)

    csv_data: list[dict] = []
    for csv_name, file_col in CSV_SOURCES.items():
        entry = process_csv(csv_name, file_col, headers, engineered_repos)
        csv_data.append(entry)

    print(f"\n{'='*60}")
    print("  Saving output files ...")
    print(f"{'='*60}")
    save_report(csv_data, REPORT_PATH)
    save_failed_report(csv_data, FAILED_PATH)
    save_analysis(csv_data, ANALYSIS_PATH)

    elapsed = perf_counter() - start
    elapsed_min, elapsed_sec = divmod(elapsed, 60)
    print(f"\n{'='*60}")
    if elapsed_min:
        print(f"  Total runtime: {int(elapsed_min)}m {elapsed_sec:.1f}s")
    else:
        print(f"  Total runtime: {elapsed_sec:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
