"""
Analyze commits.csv to identify commit messages that mention
code-clone / code-duplication related terms.

Workflow:
  1. Load commits.csv and filter rows whose repo is an engineered project.
  2. Apply a case-insensitive regex for every CLONE_TERM directly on the
     commit_message column (no HTTP fetching needed — content is inline).
  3. Collect per-commit statistics (matched terms, counts, snippets).
  4. Write detailed results to commits_clone_terms_report.json.
  5. Write a summary analysis to commits_clone_terms_analysis.json.
"""

import csv
import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AI_CONFIG_DIR  = Path("ai_config")
REPOS_CSV      = Path("ai_config/repos.csv")
COMMITS_CSV    = AI_CONFIG_DIR / "commits.csv"

RESULTS_DIR    = Path("3_commits_results")
REPORT_PATH    = RESULTS_DIR / "commits_clone_terms_report.json"
ANALYSIS_PATH  = RESULTS_DIR / "commits_clone_terms_analysis.json"

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

SNIPPET_CONTEXT = 80  # characters of context around each match


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class CommitRecord(NamedTuple):
    repo_name:        str
    commit_sha:       str
    github_link:      str
    commit_message:   str
    commit_timestamp: str
    branch:           str
    ai_tool:          str


class TermHit(NamedTuple):
    term:     str
    count:    int
    snippets: list[str]


class CommitMatchResult(NamedTuple):
    record:         CommitRecord
    hits:           list[TermHit]
    total_matches:  int
    classification: str  # "dedicated" | "mention"


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
# Step 2 – Filter commit records
# ---------------------------------------------------------------------------

def filter_commits(
    rows: list[dict],
    engineered_repos: set[str],
) -> list[CommitRecord]:
    """Keep commits whose repo is an engineered project and whose message is non-empty."""
    records: list[CommitRecord] = []
    for row in rows:
        repo_name = row.get("repo_name", "").strip()
        message = (
            row.get("commit_message", "")
            .replace("\\r\\n", "\n")
            .replace("\\n", "\n")
            .strip()
        )
        if repo_name in engineered_repos and message:
            records.append(
                CommitRecord(
                    repo_name=repo_name,
                    commit_sha=row.get("commit_sha", "").strip(),
                    github_link=row.get("github_link", "").strip(),
                    commit_message=message,
                    commit_timestamp=row.get("commit_timestamp", "").strip(),
                    branch=row.get("branch", "").strip(),
                    ai_tool=row.get("ai_tool", "").strip(),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Step 3 – Apply regex and extract statistics
# ---------------------------------------------------------------------------

def extract_snippets(text: str, term: str, context: int = SNIPPET_CONTEXT) -> list[str]:
    """Return a deduplicated list of text snippets around each match of *term*."""
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    snippets: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(text):
        start = max(0, match.start() - context)
        end   = min(len(text), match.end() + context)
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


def classify_commit(message: str, hits: list[TermHit]) -> str:
    """Return 'dedicated' if any matched term appears in the first two non-empty lines, else 'mention'."""
    non_empty = [line for line in message.splitlines() if line.strip()]
    header = "\n".join(non_empty[:2])
    for hit in hits:
        if re.search(re.escape(hit.term), header, re.IGNORECASE):
            return "dedicated"
    return "mention"


# ---------------------------------------------------------------------------
# Step 4 – Process commits.csv
# ---------------------------------------------------------------------------

def process_commits(engineered_repos: set[str]) -> dict:
    """
    Load commits.csv, filter to engineered repos, apply term search on each
    commit_message directly (no HTTP fetching required).
    Returns a dict with all data needed for both output files.
    """
    print(f"\n{'='*60}")
    print(f"  Source: {COMMITS_CSV}")
    print(f"{'='*60}")

    rows = load_csv(COMMITS_CSV)
    print(f"  Total commits: {len(rows):,}")

    records = filter_commits(rows, engineered_repos)
    print(f"  Commits from engineered repos (with message): {len(records):,}")

    results: list[CommitMatchResult] = []
    n_records = len(records)

    for idx, record in enumerate(records, start=1):
        hits = apply_regex(record.commit_message)
        if hits:
            n_hits = sum(h.count for h in hits)
            classification = classify_commit(record.commit_message, hits)
            results.append(CommitMatchResult(record=record, hits=hits, total_matches=n_hits, classification=classification))
            print(
                f"  [{idx:>{len(str(n_records))}}/{n_records}] MATCH  "
                f"{record.repo_name}  sha={record.commit_sha[:8]}  "
                f"→ {len(hits)} term(s), {n_hits} occurrence(s)"
            )

    return {
        "total_records": len(rows),
        "total_filtered": len(records),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Step 5 – Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_stats(results: list[CommitMatchResult]) -> dict:
    """Return term-commit-count mapping, classification counts, and total matched commits."""
    term_commit_counts: dict[str, int] = {t: 0 for t in CLEAN_TERMS}
    classification_counts: dict[str, int] = {"dedicated": 0, "mention": 0}
    for result in results:
        for hit in result.hits:
            term_commit_counts[hit.term] += 1
        classification_counts[result.classification] = classification_counts.get(result.classification, 0) + 1
    return {
        "total_commits_with_match": len(results),
        "classification_counts": classification_counts,
        "term_commit_counts": term_commit_counts,
        "top_terms": sorted(
            [{"term": t, "commits": c} for t, c in term_commit_counts.items() if c > 0],
            key=lambda x: x["commits"],
            reverse=True,
        ),
    }


# ---------------------------------------------------------------------------
# Step 6 – Save commits_clone_terms_report.json  (detailed)
# ---------------------------------------------------------------------------

def save_report(data: dict, report_path: Path) -> None:
    """Write one entry per matched commit with full detail."""
    output = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "total_commits":  data["total_records"],
        "total_filtered": data["total_filtered"],
        "total_matched":  len(data["results"]),
        "matches": [
            {
                "repo_name":        r.record.repo_name,
                "commit_sha":       r.record.commit_sha,
                "github_link":      r.record.github_link,
                "commit_timestamp": r.record.commit_timestamp,
                "branch":           r.record.branch,
                "ai_tool":          r.record.ai_tool,
                "total_occurrences": r.total_matches,
                "classification":    r.classification,
                "commit_message":   r.record.commit_message,
                "terms": [
                    {
                        "term":     hit.term,
                        "count":    hit.count,
                        "snippets": hit.snippets[:3],
                    }
                    for hit in r.hits
                ],
            }
            for r in data["results"]
        ],
    }

    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[SAVED] Detailed report  → {report_path}")


# ---------------------------------------------------------------------------
# Step 7 – Save commits_clone_terms_analysis.json  (summary)
# ---------------------------------------------------------------------------

def save_analysis(data: dict, analysis_path: Path) -> None:
    """Write a summary analysis JSON with term rankings and totals."""
    stats = aggregate_stats(data["results"])

    # Breakdown by ai_tool
    tool_counts: dict[str, int] = {}
    for r in data["results"]:
        tool = r.record.ai_tool or "unknown"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    # Breakdown by repo
    repo_counts: dict[str, int] = {}
    for r in data["results"]:
        repo_counts[r.record.repo_name] = repo_counts.get(r.record.repo_name, 0) + 1

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_commits_in_csv":    data["total_records"],
            "commits_from_engineered": data["total_filtered"],
            "commits_with_match":      stats["total_commits_with_match"],
            "classification_counts":   stats["classification_counts"],
            "top_terms":               stats["top_terms"],
            "matches_by_ai_tool": sorted(
                [{"ai_tool": t, "commits": c} for t, c in tool_counts.items()],
                key=lambda x: x["commits"],
                reverse=True,
            ),
            "matches_by_repo": sorted(
                [{"repo_name": r, "commits": c} for r, c in repo_counts.items()],
                key=lambda x: x["commits"],
                reverse=True,
            ),
        },
    }

    analysis_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] Analysis summary → {analysis_path}")


# ---------------------------------------------------------------------------
# Startup report
# ---------------------------------------------------------------------------

def print_startup_report(engineered_repos: set[str]) -> None:
    rows    = load_csv(COMMITS_CSV)
    total   = len(rows)
    filtered = sum(
        1 for row in rows
        if row.get("repo_name", "").strip() in engineered_repos
        and row.get("commit_message", "").strip()
    )

    print("=" * 60)
    print("  Pre-run summary")
    print("=" * 60)
    print(f"  Total commits in CSV:              {total:>10,}")
    print(f"  Commits from engineered repos:     {filtered:>10,}")
    print(f"  Terms to search:                   {len(CLEAN_TERMS):>10,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    start = perf_counter()

    engineered_repos = load_engineered_repos(REPOS_CSV)
    print(f"  Engineered repos loaded: {len(engineered_repos):,} from {REPOS_CSV}")

    print_startup_report(engineered_repos)

    RESULTS_DIR.mkdir(exist_ok=True)

    data = process_commits(engineered_repos)

    print(f"\n{'='*60}")
    print("  Saving output files ...")
    print(f"{'='*60}")
    save_report(data, REPORT_PATH)
    save_analysis(data, ANALYSIS_PATH)

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
