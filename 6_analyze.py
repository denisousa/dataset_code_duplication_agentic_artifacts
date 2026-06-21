from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "4_analyze_results"

MD_ANALYSIS_PATH = ROOT_DIR / "1_configurations_artifacts_results" / "clone_terms_analysis.json"
MD_REPORT_PATH = ROOT_DIR / "1_configurations_artifacts_results" / "clone_terms_report.json"
COMMITS_ANALYSIS_PATH = ROOT_DIR / "2_commits_results" / "commits_clone_terms_analysis.json"
COMMITS_REPORT_PATH = ROOT_DIR / "2_commits_results" / "commits_clone_terms_report.json"
PRS_RESULTS_PATH = ROOT_DIR / "3_prs_results" / "search_prs_results.json"
PRS_CHECKPOINT_PATH = ROOT_DIR / "3_prs_results" / "search_prs_checkpoint.json"
TARGET_AGENTS = ["Claude", "Gemini", "Cursor", "Copilot", "Codex", "CodeRabbit", "Others"]
COMMIT_AGENT_BUCKETS = [*TARGET_AGENTS, "more than one"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def normalize_term(term: str) -> str:
    return " ".join(term.strip().split()).lower()


def choose_display_term(display_counts: Counter[str]) -> str:
    return sorted(display_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def count_terms_in_md(report_data: dict[str, Any]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    counts: Counter[str] = Counter()
    display_variants: dict[str, Counter[str]] = {}

    for source in report_data.get("sources", []):
        for match in source.get("matches", []):
            for term_entry in match.get("terms", []):
                term = term_entry.get("term")
                if term:
                    normalized_term = normalize_term(term)
                    counts[normalized_term] += 1
                    display_variants.setdefault(normalized_term, Counter())[term] += 1

    return counts, display_variants


def count_terms_in_commits(report_data: dict[str, Any]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    counts: Counter[str] = Counter()
    display_variants: dict[str, Counter[str]] = {}

    for match in report_data.get("matches", []):
        for term_entry in match.get("terms", []):
            term = term_entry.get("term")
            if term:
                normalized_term = normalize_term(term)
                counts[normalized_term] += 1
                display_variants.setdefault(normalized_term, Counter())[term] += 1

    return counts, display_variants


def count_terms_in_prs(prs_data: dict[str, list[dict[str, Any]]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    counts: Counter[str] = Counter()
    display_variants: dict[str, Counter[str]] = {}

    for repo_prs in prs_data.values():
        for pr_entry in repo_prs:
            unique_terms = set(pr_entry.get("matched_terms") or [])

            if not unique_terms:
                unique_terms = {
                    match.get("term")
                    for match in pr_entry.get("matches", [])
                    if match.get("term")
                }

            for term in unique_terms:
                normalized_term = normalize_term(term)
                counts[normalized_term] += 1
                display_variants.setdefault(normalized_term, Counter())[term] += 1

    return counts, display_variants


def count_terms_in_pr_comments(prs_data: dict[str, list[dict[str, Any]]]) -> tuple[Counter[str], dict[str, Counter[str]]]:
    counts: Counter[str] = Counter()
    display_variants: dict[str, Counter[str]] = {}

    for repo_name, repo_prs in prs_data.items():
        for pr_entry in repo_prs:
            seen_comment_term_keys: set[tuple[str, int, str | None, str | None, str | None, str]] = set()

            for match in pr_entry.get("matches", []):
                location = match.get("location")
                author = match.get("author")
                author_type = match.get("author_type")
                term = match.get("term")

                if location in {"PR title", "PR body"} or not term:
                    continue

                normalized_term = normalize_term(term)
                comment_term_key = (
                    repo_name,
                    int(pr_entry.get("number") or 0),
                    location,
                    author,
                    author_type,
                    normalized_term,
                )
                if comment_term_key in seen_comment_term_keys:
                    continue

                seen_comment_term_keys.add(comment_term_key)
                counts[normalized_term] += 1
                display_variants.setdefault(normalized_term, Counter())[term] += 1

    return counts, display_variants


def build_top_terms_rows(
    md_term_counts: Counter[str],
    commit_term_counts: Counter[str],
    pr_term_counts: Counter[str],
    display_variants: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_terms = sorted(
        set(md_term_counts) | set(commit_term_counts) | set(pr_term_counts)
    )

    for term in all_terms:
        md_count = md_term_counts.get(term, 0)
        commit_count = commit_term_counts.get(term, 0)
        pr_count = pr_term_counts.get(term, 0)
        total = md_count + commit_count + pr_count

        rows.append(
            {
                "term": choose_display_term(display_variants[term]),
                "configuration_artifacts": md_count,
                "commits": commit_count,
                "prs": pr_count,
                "total": total,
            }
        )

    rows.sort(key=lambda row: (-row["total"], -row["configuration_artifacts"], row["term"]))
    return rows


def build_commit_distribution_rows(commits_analysis: dict[str, Any]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    total_commits_in_csv = int(commits_analysis.get("summary", {}).get("total_commits_in_csv") or 0)
    total_commits_with_match = int(commits_analysis.get("summary", {}).get("commits_with_match") or 0)

    rows.append(
        {
            "term": "total_commits",
            "commits": total_commits_in_csv,
        }
    )
    rows.append(
        {
            "term": "total_matched_commits",
            "commits": total_commits_with_match,
        }
    )

    for entry in commits_analysis.get("summary", {}).get("top_terms", []):
        rows.append(
            {
                "term": str(entry.get("term", "")),
                "commits": int(entry.get("commits") or 0),
            }
        )

    return rows


def build_commit_agent_rows(commits_report: dict[str, Any]) -> list[dict[str, int | str]]:
    counts_by_agent: Counter[str] = Counter()

    for match in commits_report.get("matches", []):
        ai_tool_value = str(match.get("ai_tool") or "")
        mapped_agents = {
            classify_target_agent(tool_name.strip())
            for tool_name in ai_tool_value.split(",")
            if tool_name.strip()
        }

        mapped_agents.discard(None)
        if len(mapped_agents) > 1:
            counts_by_agent["more than one"] += 1
        elif mapped_agents:
            counts_by_agent[next(iter(mapped_agents))] += 1

    rows = [
        {
            "agent": agent_name,
            "commit_count": counts_by_agent.get(agent_name, 0),
        }
        for agent_name in COMMIT_AGENT_BUCKETS
    ]
    rows.sort(
        key=lambda row: (
            str(row["agent"]) == "Others",
            str(row["agent"]) == "more than one",
            -int(row["commit_count"]),
            str(row["agent"]),
        )
    )
    rows.append(
        {
            "agent": "total",
            "commit_count": sum(int(row["commit_count"]) for row in rows),
        }
    )
    return rows


def build_cross_source_term_rows(
    md_term_counts: Counter[str],
    commit_term_counts: Counter[str],
    pr_comment_term_counts: Counter[str],
    display_variants: dict[str, Counter[str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_terms = sorted(
        set(md_term_counts) | set(commit_term_counts) | set(pr_comment_term_counts)
    )

    for term in all_terms:
        md_count = md_term_counts.get(term, 0)
        commit_count = commit_term_counts.get(term, 0)
        pr_comment_count = pr_comment_term_counts.get(term, 0)
        total = md_count + commit_count + pr_comment_count

        rows.append(
            {
                "term": choose_display_term(display_variants[term]),
                "configuration_artifacts": md_count,
                "commits": commit_count,
                "pr_comments": pr_comment_count,
                "total": total,
            }
        )

    rows.sort(key=lambda row: (-row["total"], -row["commits"], row["term"]))
    return rows


def build_classification_rows(
    md_analysis: dict[str, Any],
    commits_analysis: dict[str, Any],
    prs_data: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    md_counts = md_analysis["global_summary"]["classification_counts"]
    commit_counts = commits_analysis["summary"]["classification_counts"]

    pr_dedicated = 0
    pr_mention_like = 0

    for repo_prs in prs_data.values():
        for pr_entry in repo_prs:
            discussion_type = pr_entry.get("discussion_type")

            if discussion_type == "dedicated":
                pr_dedicated += 1
            elif discussion_type in {"mention", "partial"}:
                pr_mention_like += 1

    rows = [
        {
            "source": "configuration_artifacts",
            "dedicated": md_counts.get("dedicated", 0),
            "mention_including_partial": md_counts.get("mention", 0)
            + md_counts.get("partial", 0),
            "total_matched": md_analysis["global_summary"].get("total_md_files_with_match", 0),
        },
        {
            "source": "commits",
            "dedicated": commit_counts.get("dedicated", 0),
            "mention_including_partial": commit_counts.get("mention", 0)
            + commit_counts.get("partial", 0),
            "total_matched": commits_analysis["summary"].get("commits_with_match", 0),
        },
        {
            "source": "prs",
            "dedicated": pr_dedicated,
            "mention_including_partial": pr_mention_like,
            "total_matched": pr_dedicated + pr_mention_like,
        },
    ]

    rows.append(
        {
            "source": "total",
            "dedicated": sum(row["dedicated"] for row in rows),
            "mention_including_partial": sum(
                row["mention_including_partial"] for row in rows
            ),
            "total_matched": sum(row["total_matched"] for row in rows),
        }
    )

    return rows


def build_md_breakdown_row(md_analysis: dict[str, Any]) -> dict[str, int]:
    counts = md_analysis["global_summary"]["classification_counts"]
    dedicated = counts.get("dedicated", 0)
    partial = counts.get("partial", 0)
    mention = counts.get("mention", 0)

    return {
        "dedicated": dedicated,
        "partial": partial,
        "mention": mention,
        "total": dedicated + partial + mention,
    }


def build_md_source_match_rows(md_analysis: dict[str, Any]) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []

    for entry in md_analysis.get("per_csv", []):
        csv_name = str(entry.get("csv", ""))
        source_name = csv_name.removesuffix(".csv") if csv_name else "unknown"
        rows.append(
            {
                "source_type": source_name,
                "matched_files": int(entry.get("md_files_with_match") or 0),
                "total_files": int(entry.get("total_md_files") or 0),
            }
        )

    rows.sort(key=lambda row: (-int(row["matched_files"]), str(row["source_type"])))
    rows.append(
        {
            "source_type": "total",
            "matched_files": int(md_analysis["global_summary"].get("total_md_files_with_match") or 0),
            "total_files": int(md_analysis["global_summary"].get("total_md_files_across_all_csvs") or 0),
        }
    )
    return rows


def build_dataset_totals_rows(
    md_analysis: dict[str, Any],
    md_report: dict[str, Any],
    commits_analysis: dict[str, Any],
    commits_report: dict[str, Any],
    prs_results: dict[str, list[dict[str, Any]]],
    prs_checkpoint: dict[str, Any],
) -> list[dict[str, int | str]]:
    total_prs = 0
    matched_prs = 0
    md_repos: set[str] = set()
    commit_repos: set[str] = set()

    for repo_state in prs_checkpoint.get("repos", {}).values():
        processed_prs = repo_state.get("processed_prs") or {}
        total_prs += len(processed_prs)
        matched_prs += sum(1 for value in processed_prs.values() if value is not None)

    for source in md_report.get("sources", []):
        for match in source.get("matches", []):
            repo_name = str(match.get("repo_name") or "").strip()
            if repo_name:
                md_repos.add(repo_name)

    for match in commits_report.get("matches", []):
        repo_name = str(match.get("repo_name") or "").strip()
        if repo_name:
            commit_repos.add(repo_name)

    pr_repos = set(prs_results.keys())

    rows = [
        {
            "source": "configuration_artifacts",
            "total_items": int(md_analysis.get("global_summary", {}).get("total_md_files_across_all_csvs") or 0),
            "matched_items": int(md_analysis.get("global_summary", {}).get("total_md_files_with_match") or 0),
            "unique_repositories": len(md_repos),
        },
        {
            "source": "commits",
            "total_items": int(commits_analysis.get("summary", {}).get("total_commits_in_csv") or 0),
            "matched_items": int(commits_analysis.get("summary", {}).get("commits_with_match") or 0),
            "unique_repositories": len(commit_repos),
        },
        {
            "source": "prs",
            "total_items": total_prs,
            "matched_items": matched_prs,
            "unique_repositories": len(pr_repos),
        },
    ]
    rows.append(
        {
            "source": "total",
            "total_items": sum(int(row["total_items"]) for row in rows),
            "matched_items": sum(int(row["matched_items"]) for row in rows),
            "unique_repositories": len(md_repos | commit_repos | pr_repos),
        }
    )
    return rows


def build_collection_summary_rows(
    md_report: dict[str, Any],
    commits_report: dict[str, Any],
    prs_results: dict[str, list[dict[str, Any]]],
    prs_checkpoint: dict[str, Any],
) -> list[dict[str, int | str]]:
    md_repos: set[str] = set()
    commit_repos: set[str] = set()

    matched_configuration_artifacts = 0
    for source in md_report.get("sources", []):
        matched_configuration_artifacts += len(source.get("matches", []))
        for match in source.get("matches", []):
            repo_name = str(match.get("repo_name") or "").strip()
            if repo_name:
                md_repos.add(repo_name)

    matched_commits = len(commits_report.get("matches", []))
    for match in commits_report.get("matches", []):
        repo_name = str(match.get("repo_name") or "").strip()
        if repo_name:
            commit_repos.add(repo_name)

    pr_repos = set(prs_results.keys())
    matched_prs = sum(len(repo_prs) for repo_prs in prs_results.values())
    matched_pr_comments = sum(
        int(pr_entry.get("matched_comments_count") or 0)
        for repo_prs in prs_results.values()
        for pr_entry in repo_prs
    )

    return [
        {
            "metric": "unique_repositories_with_configuration_artifacts_duplication",
            "value": len(md_repos),
        },
        {
            "metric": "unique_repositories_with_agent_commits",
            "value": len(commit_repos),
        },
        {
            "metric": "unique_repositories_with_prs",
            "value": len(pr_repos),
        },
        {
            "metric": "unique_repositories_total",
            "value": len(md_repos | commit_repos | pr_repos),
        },
        {
            "metric": "matched_configuration_artifacts",
            "value": matched_configuration_artifacts,
        },
        {
            "metric": "matched_commits",
            "value": matched_commits,
        },
        {
            "metric": "matched_prs",
            "value": matched_prs,
        },
        {
            "metric": "matched_pr_comments",
            "value": matched_pr_comments,
        },
        {
            "metric": "total_matched_artifacts",
            "value": matched_configuration_artifacts + matched_commits + matched_prs,
        },
    ]


def count_pr_comments_by_author_type(pr_entry: dict[str, Any]) -> tuple[int, int, int]:
    matched_comments_count = int(pr_entry.get("matched_comments_count") or 0)
    commenter_types = pr_entry.get("commenter_types")

    if matched_comments_count == 0:
        return 0, 0, 0

    if commenter_types == "human":
        return matched_comments_count, 0, 0

    if commenter_types == "bot":
        return 0, matched_comments_count, 0

    unique_comment_keys: set[tuple[str | None, str | None, str | None]] = set()
    human_comments = 0
    agent_comments = 0

    for match in pr_entry.get("matches", []):
        location = match.get("location")
        author = match.get("author")
        author_type = match.get("author_type")

        if location in {"PR title", "PR body"} or not author_type:
            continue

        comment_key = (location, author, author_type)
        if comment_key in unique_comment_keys:
            continue

        unique_comment_keys.add(comment_key)
        if author_type == "human":
            human_comments += 1
        elif author_type == "bot":
            agent_comments += 1

    unresolved_comments = max(matched_comments_count - human_comments - agent_comments, 0)
    return human_comments, agent_comments, unresolved_comments


def classify_target_agent(author: str | None) -> str | None:
    if not author:
        return None

    author_lower = author.lower()
    for agent_name in TARGET_AGENTS:
        if agent_name == "Others":
            continue
        if agent_name.lower() in author_lower:
            return agent_name

    return "Others"


def build_pr_creator_comment_rows(prs_data: dict[str, list[dict[str, Any]]]) -> list[dict[str, int | str]]:
    buckets: dict[str, dict[str, int | str]] = {
        "human": {
            "pr_creator_type": "human",
            "pr_count": 0,
            "human_comments": 0,
            "agent_comments": 0,
            "unresolved_comments": 0,
            "total_comments": 0,
        },
        "agent": {
            "pr_creator_type": "agent",
            "pr_count": 0,
            "human_comments": 0,
            "agent_comments": 0,
            "unresolved_comments": 0,
            "total_comments": 0,
        },
    }

    for repo_prs in prs_data.values():
        for pr_entry in repo_prs:
            actor_type = pr_entry.get("actor_type")
            if actor_type not in buckets:
                continue

            human_comments, agent_comments, unresolved_comments = count_pr_comments_by_author_type(pr_entry)
            total_comments = int(pr_entry.get("matched_comments_count") or 0)
            bucket = buckets[actor_type]
            bucket["pr_count"] += 1
            bucket["human_comments"] += human_comments
            bucket["agent_comments"] += agent_comments
            bucket["unresolved_comments"] += unresolved_comments
            bucket["total_comments"] += total_comments

    rows = [buckets["human"], buckets["agent"]]
    rows.append(
        {
            "pr_creator_type": "total",
            "pr_count": sum(int(row["pr_count"]) for row in rows),
            "human_comments": sum(int(row["human_comments"]) for row in rows),
            "agent_comments": sum(int(row["agent_comments"]) for row in rows),
            "unresolved_comments": sum(int(row["unresolved_comments"]) for row in rows),
            "total_comments": sum(int(row["total_comments"]) for row in rows),
        }
    )
    return rows


def build_comment_agent_count_rows(prs_data: dict[str, list[dict[str, Any]]]) -> list[dict[str, int | str]]:
    rows_by_agent: dict[str, dict[str, int | str]] = {
        agent_name: {
            "agent": agent_name,
            "comment_count": 0,
        }
        for agent_name in TARGET_AGENTS
    }

    for repo_prs in prs_data.values():
        for pr_entry in repo_prs:
            commenter_types = pr_entry.get("commenter_types")
            matched_comments_count = int(pr_entry.get("matched_comments_count") or 0)

            if matched_comments_count == 0:
                continue

            bot_matches = [
                match
                for match in pr_entry.get("matches", [])
                if match.get("location") not in {"PR title", "PR body"}
                and match.get("author")
                and match.get("author_type") == "bot"
            ]

            if not bot_matches:
                continue

            if commenter_types == "bot":
                agent_to_keys: dict[str, set[tuple[str | None, str | None]]] = {}
                for match in bot_matches:
                    agent_name = classify_target_agent(match.get("author"))
                    if not agent_name:
                        continue

                    comment_key = (match.get("location"), match.get("author"))
                    agent_to_keys.setdefault(agent_name, set()).add(comment_key)

                if len(agent_to_keys) == 1:
                    agent_name = next(iter(agent_to_keys))
                    rows_by_agent[agent_name]["comment_count"] += matched_comments_count
                else:
                    for agent_name, comment_keys in agent_to_keys.items():
                        rows_by_agent[agent_name]["comment_count"] += len(comment_keys)
                continue

            seen_comment_keys: set[tuple[str | None, str | None, str | None]] = set()
            for match in bot_matches:
                agent_name = classify_target_agent(match.get("author"))
                if not agent_name:
                    continue

                comment_key = (
                    match.get("location"),
                    match.get("author"),
                    match.get("author_type"),
                )
                if comment_key in seen_comment_keys:
                    continue

                seen_comment_keys.add(comment_key)
                rows_by_agent[agent_name]["comment_count"] += 1

    rows = list(rows_by_agent.values())
    rows.sort(
        key=lambda row: (
            str(row["agent"]) == "Others",
            -int(row["comment_count"]),
            str(row["agent"]),
        )
    )
    rows.append(
        {
            "agent": "total",
            "comment_count": sum(int(row["comment_count"]) for row in rows),
        }
    )
    return rows


def build_agent_artifact_rows(
    md_report: dict[str, Any],
    commits_analysis: dict[str, Any],
    comment_agent_count_rows: list[dict[str, int | str]],
) -> list[dict[str, int | str]]:
    rows_by_agent: dict[str, dict[str, int | str]] = {
        agent_name: {
            "agent": agent_name,
            "configuration_artifacts": 0,
            "commits": 0,
            "pr_comments": 0,
            "total": 0,
        }
        for agent_name in COMMIT_AGENT_BUCKETS
    }

    for source in md_report.get("sources", []):
        for match in source.get("matches", []):
            agent_name = classify_target_agent(
                f'{match.get("file_path", "")} {match.get("file_name", "")}'
            )
            if not agent_name:
                continue
            rows_by_agent[agent_name]["configuration_artifacts"] += 1

    for entry in commits_analysis.get("summary", {}).get("matches_by_ai_tool", []):
        commit_count = int(entry.get("commits") or 0)
        ai_tool_value = str(entry.get("ai_tool") or "")
        mapped_agents = {
            classify_target_agent(tool_name.strip())
            for tool_name in ai_tool_value.split(",")
            if tool_name.strip()
        }

        mapped_agents.discard(None)
        if len(mapped_agents) > 1:
            rows_by_agent["more than one"]["commits"] += commit_count
        elif mapped_agents:
            rows_by_agent[next(iter(mapped_agents))]["commits"] += commit_count

    for row in comment_agent_count_rows:
        agent_name = str(row.get("agent") or "")
        if agent_name == "total" or agent_name not in rows_by_agent:
            continue
        rows_by_agent[agent_name]["pr_comments"] = int(row.get("comment_count") or 0)

    rows = [
        row for row in rows_by_agent.values()
        if str(row["agent"]) != "more than one"
    ]
    for row in rows:
        row["total"] = (
            int(row["configuration_artifacts"])
            + int(row["commits"])
            + int(row["pr_comments"])
        )

    rows.sort(
        key=lambda row: (
            str(row["agent"]) == "Others",
            -int(row["total"]),
            str(row["agent"]),
        )
    )
    rows.append(
        {
            "agent": "total",
            "configuration_artifacts": sum(int(row["configuration_artifacts"]) for row in rows),
            "commits": sum(int(row["commits"]) for row in rows),
            "pr_comments": sum(int(row["pr_comments"]) for row in rows),
            "total": sum(int(row["total"]) for row in rows),
        }
    )
    return rows


def format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "(no data)"

    widths = []
    for key, header in columns:
        max_value_width = max(len(str(row.get(key, ""))) for row in rows)
        widths.append(max(len(header), max_value_width))

    header_line = " | ".join(
        header.ljust(width) for (_, header), width in zip(columns, widths)
    )
    separator_line = "-+-".join("-" * width for width in widths)

    data_lines = []
    for row in rows:
        parts = []
        for (key, _), width in zip(columns, widths):
            value = row.get(key, "")
            if isinstance(value, int):
                parts.append(str(value).rjust(width))
            else:
                parts.append(str(value).ljust(width))
        data_lines.append(" | ".join(parts))

    return "\n".join([header_line, separator_line, *data_lines])


def validate_md_total(md_analysis: dict[str, Any], md_breakdown: dict[str, int]) -> None:
    expected_total = md_analysis["global_summary"].get("total_md_files_with_match", 0)
    actual_total = md_breakdown["total"]

    if expected_total != actual_total:
        raise ValueError(
            "MD classification totals do not match the reported matched-file total: "
            f"expected {expected_total}, got {actual_total}."
        )


def build_text_report(
    md_source_match_rows: list[dict[str, int | str]],
    commit_distribution_rows: list[dict[str, int | str]],
    commit_agent_rows: list[dict[str, int | str]],
    collection_summary_rows: list[dict[str, int | str]],
    cross_source_term_rows: list[dict[str, Any]],
    top_terms_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
    md_breakdown_row: dict[str, int],
    pr_creator_comment_rows: list[dict[str, int | str]],
    comment_agent_count_rows: list[dict[str, int | str]],
    agent_artifact_rows: list[dict[str, int | str]],
    dataset_totals_rows: list[dict[str, int | str]],
) -> str:
    sections = [
        "MATCHED CONFIGURATION FILES",
        format_table(
            md_source_match_rows,
            [
                ("source_type", "source_type"),
                ("matched_files", "matched_files"),
                ("total_files", "total_files"),
            ],
        ),
        "",
        "COMMITS DISTRIBUTION",
        format_table(
            commit_distribution_rows,
            [
                ("term", "term"),
                ("commits", "commits"),
            ],
        ),
        "",
        "COMMITS AUTHORED BY AGENTS",
        format_table(
            commit_agent_rows,
            [
                ("agent", "agent"),
                ("commit_count", "commit_count"),
            ],
        ),
        "",
        "COLLECTION SUMMARY",
        format_table(
            collection_summary_rows,
            [
                ("metric", "metric"),
                ("value", "value"),
            ],
        ),
        "",
        "CONFIGURATION ARTIFACTS, COMMITS, AND PR COMMENTS",
        format_table(
            cross_source_term_rows,
            [
                ("term", "term"),
                ("configuration_artifacts", "configuration_artifacts"),
                ("commits", "commits"),
                ("pr_comments", "pr_comments"),
                ("total", "total"),
            ],
        ),
        "",
        "PR CREATOR VS COMMENT AUTHOR COUNTS",
        format_table(
            pr_creator_comment_rows,
            [
                ("pr_creator_type", "pr_creator_type"),
                ("pr_count", "pr_count_creator"),
                ("human_comments", "human_comments"),
                ("agent_comments", "agent_comments"),
                ("unresolved_comments", "unresolved_comments"),
                ("total_comments", "total_comments"),
            ],
        ),
        "",
        "COMMENT AGENTS COUNTS",
        format_table(
            comment_agent_count_rows,
            [
                ("agent", "agent"),
                ("comment_count", "comment_count"),
            ],
        ),
        "",
        "AGENT ARTIFACT COUNTS",
        format_table(
            agent_artifact_rows,
            [
                ("agent", "agent"),
                ("configuration_artifacts", "configuration_artifacts"),
                ("commits", "commits"),
                ("pr_comments", "pr_comments"),
                ("total", "total"),
            ],
        ),
        "",
        "DATASET TOTALS",
        format_table(
            dataset_totals_rows,
            [
                ("source", "source"),
                ("total_items", "total_items"),
                ("matched_items", "matched_items"),
                ("unique_repositories", "unique_repositories"),
            ],
        ),
        "",
        "TOP CLONE TERMS ACROSS CONFIGURATION ARTIFACTS, COMMITS, AND PRS",
        format_table(
            top_terms_rows,
            [
                ("term", "term"),
                ("configuration_artifacts", "configuration_artifacts"),
                ("commits", "commits"),
                ("prs", "prs"),
                ("total", "total"),
            ],
        ),
        "",
        "DEDICATED VS MENTION-LIKE COUNTS",
        format_table(
            classification_rows,
            [
                ("source", "source"),
                ("dedicated", "dedicated"),
                ("mention_including_partial", "mention_including_partial"),
                ("total_matched", "total_matched"),
            ],
        ),
        "",
        "CONFIGURATION ARTIFACTS BREAKDOWN",
        format_table(
            [md_breakdown_row],
            [
                ("dedicated", "dedicated"),
                ("partial", "partial"),
                ("mention", "mention"),
                ("total", "total"),
            ],
        ),
    ]

    return "\n".join(sections)


def escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def intersects(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def estimate_text_box(term: str, font_size: float, center_x: float, center_y: float) -> tuple[float, float, float, float]:
    width = max(font_size * 0.58 * len(term), font_size)
    height = font_size * 0.95
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return x1, y1, x2, y2


def build_wordcloud_svg(top_terms_rows: list[dict[str, Any]]) -> str:
    canvas_width = 1400
    canvas_height = 900
    padding = 28
    content_left = 90
    content_top = 190
    content_right = 1310
    content_bottom = 790
    content_width = content_right - content_left
    content_height = content_bottom - content_top
    background_color = "#f4efe7"
    accent_color = "#d97745"
    palette = ["#183a52", "#2c6e91", "#bb4d3e", "#e19a3e", "#2f7d66", "#6b4e3d"]
    rows = [row for row in top_terms_rows if row["total"] > 0]

    if not rows:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
            f'viewBox="0 0 {canvas_width} {canvas_height}">'
            f'<rect width="100%" height="100%" fill="{background_color}" />'
            '<text x="50%" y="50%" text-anchor="middle" font-size="36" '
            'font-family="DejaVu Sans, Arial, sans-serif" fill="#333333">No data</text>'
            '</svg>'
        )

    max_total = max(row["total"] for row in rows)
    min_total = min(row["total"] for row in rows)
    random_generator = random.Random(42)
    placed_boxes: list[tuple[float, float, float, float]] = []
    placed_words: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if max_total == min_total:
            scale = 1.0
        else:
            scale = (row["total"] - min_total) / (max_total - min_total)

        font_size = 24 + scale * 92
        angle = 0
        found_position = False

        for step in range(1200):
            spiral = step / 12
            theta = step * 0.47
            radius_x = spiral * 8.5
            radius_y = spiral * 5.4
            center_x = content_left + content_width / 2 + math.cos(theta) * radius_x
            center_y = content_top + content_height / 2 + math.sin(theta) * radius_y

            jitter_x = random_generator.uniform(-8, 8)
            jitter_y = random_generator.uniform(-6, 6)
            center_x += jitter_x
            center_y += jitter_y

            box = estimate_text_box(row["term"], font_size, center_x, center_y)
            x1, y1, x2, y2 = box

            if (
                x1 < content_left + padding
                or y1 < content_top + padding
                or x2 > content_right - padding
                or y2 > content_bottom - padding
            ):
                continue

            if any(intersects(box, existing_box) for existing_box in placed_boxes):
                continue

            placed_boxes.append(box)
            placed_words.append(
                {
                    "term": row["term"],
                    "font_size": round(font_size, 2),
                    "x": round(center_x, 2),
                    "y": round(center_y, 2),
                    "fill": palette[index % len(palette)],
                    "rotation": angle,
                    "total": row["total"],
                }
            )
            found_position = True
            break

        if not found_position:
            center_x = content_left + content_width / 2
            center_y = content_top + content_height / 2
            box = estimate_text_box(row["term"], font_size, center_x, center_y)
            placed_boxes.append(box)
            placed_words.append(
                {
                    "term": row["term"],
                    "font_size": round(font_size, 2),
                    "x": round(center_x, 2),
                    "y": round(center_y, 2),
                    "fill": palette[index % len(palette)],
                    "rotation": 0,
                    "total": row["total"],
                }
            )

    hero_term = rows[0]["term"]
    hero_total = rows[0]["total"]
    footer_terms = ", ".join(row["term"] for row in rows[:4])

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
        '<defs>'
        '<linearGradient id="bg-gradient" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#fbf7f0" />'
        '<stop offset="100%" stop-color="#efe6d8" />'
        '</linearGradient>'
        '<linearGradient id="card-gradient" x1="0%" y1="0%" x2="0%" y2="100%">'
        '<stop offset="0%" stop-color="#fffdf8" />'
        '<stop offset="100%" stop-color="#f8f1e6" />'
        '</linearGradient>'
        '<filter id="card-shadow" x="-20%" y="-20%" width="140%" height="160%">'
        '<feDropShadow dx="0" dy="18" stdDeviation="24" flood-color="#9b8b79" flood-opacity="0.18" />'
        '</filter>'
        '<filter id="text-shadow" x="-20%" y="-20%" width="140%" height="140%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#ffffff" flood-opacity="0.65" />'
        '</filter>'
        '<radialGradient id="orb-left" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#f7d9c7" stop-opacity="0.9" />'
        '<stop offset="100%" stop-color="#f7d9c7" stop-opacity="0" />'
        '</radialGradient>'
        '<radialGradient id="orb-right" cx="50%" cy="50%" r="50%">'
        '<stop offset="0%" stop-color="#d7e8ef" stop-opacity="0.95" />'
        '<stop offset="100%" stop-color="#d7e8ef" stop-opacity="0" />'
        '</radialGradient>'
        '</defs>',
        '<rect width="100%" height="100%" fill="url(#bg-gradient)" />',
        '<circle cx="170" cy="170" r="170" fill="url(#orb-left)" />',
        '<circle cx="1220" cy="710" r="220" fill="url(#orb-right)" />',
        '<rect x="58" y="42" width="1284" height="810" rx="34" fill="url(#card-gradient)" filter="url(#card-shadow)" />',
        f'<rect x="58" y="42" width="1284" height="810" rx="34" fill="none" stroke="{accent_color}" stroke-opacity="0.16" />',
        f'<text x="110" y="98" font-size="15" font-weight="700" letter-spacing="1.8" font-family="DejaVu Sans, Arial, sans-serif" fill="{accent_color}">DATASET SUMMARY</text>',
        '<text x="110" y="142" font-size="38" font-weight="700" font-family="DejaVu Sans, Arial, sans-serif" fill="#1c242b">Clone Terms Word Cloud</text>',
        '<text x="110" y="176" font-size="18" font-family="DejaVu Sans, Arial, sans-serif" fill="#56616b">Word size reflects total occurrences across markdown files, commits, and pull requests.</text>',
        f'<rect x="1040" y="94" width="210" height="66" rx="18" fill="#fff6ea" stroke="{accent_color}" stroke-opacity="0.24" />',
        f'<text x="1062" y="120" font-size="13" font-weight="700" letter-spacing="1.2" font-family="DejaVu Sans, Arial, sans-serif" fill="{accent_color}">TOP TERM</text>',
        f'<text x="1062" y="147" font-size="20" font-weight="700" font-family="DejaVu Sans, Arial, sans-serif" fill="#1c242b">{escape_xml(hero_term)}</text>',
        f'<text x="1230" y="147" text-anchor="end" font-size="22" font-weight="700" font-family="DejaVu Sans, Arial, sans-serif" fill="#1c242b">{hero_total}</text>',
        '<rect x="92" y="190" width="1220" height="602" rx="28" fill="#fffdfa" stroke="#d8cfc2" stroke-opacity="0.7" />',
    ]

    for word in placed_words:
        transform = f'rotate({word["rotation"]} {word["x"]} {word["y"]})'
        opacity = 0.88 if word["total"] < hero_total * 0.15 else 0.98
        elements.append(
            f'<text x="{word["x"]}" y="{word["y"]}" text-anchor="middle" dominant-baseline="middle" '
            f'font-size="{word["font_size"]}" font-family="DejaVu Sans, Arial, sans-serif" '
            f'font-weight="700" letter-spacing="-0.4" fill="{word["fill"]}" fill-opacity="{opacity}" '
            f'stroke="#fffaf3" stroke-width="1.35" paint-order="stroke fill" filter="url(#text-shadow)" '
            f'transform="{transform}">{escape_xml(word["term"])}</text>'
        )

    elements.extend(
        [
            f'<text x="110" y="828" font-size="15" font-family="DejaVu Sans, Arial, sans-serif" fill="#67727c">Most visible terms: {escape_xml(footer_terms)}</text>',
            '<text x="1292" y="828" text-anchor="end" font-size="14" font-family="DejaVu Sans, Arial, sans-serif" fill="#8a7767">Generated by 5_analyze.py</text>',
        ]
    )

    elements.append('</svg>')
    return "".join(elements)


def write_outputs(
    md_source_match_rows: list[dict[str, int | str]],
    commit_distribution_rows: list[dict[str, int | str]],
    commit_agent_rows: list[dict[str, int | str]],
    collection_summary_rows: list[dict[str, int | str]],
    cross_source_term_rows: list[dict[str, Any]],
    top_terms_rows: list[dict[str, Any]],
    classification_rows: list[dict[str, Any]],
    md_breakdown_row: dict[str, int],
    pr_creator_comment_rows: list[dict[str, int | str]],
    comment_agent_count_rows: list[dict[str, int | str]],
    agent_artifact_rows: list[dict[str, int | str]],
    dataset_totals_rows: list[dict[str, int | str]],
) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    summary_payload = {
        "matched_configuration_files": md_source_match_rows,
        "commit_term_distribution": commit_distribution_rows,
        "commit_agent_counts": commit_agent_rows,
        "collection_summary": collection_summary_rows,
        "configuration_artifacts_commits_pr_comments": cross_source_term_rows,
        "top_clone_terms": top_terms_rows,
        "dedicated_vs_mention_like": classification_rows,
        "configuration_artifacts_breakdown": md_breakdown_row,
        "pr_creator_comment_author_counts": pr_creator_comment_rows,
        "comment_agents_counts": comment_agent_count_rows,
        "agent_artifact_counts": agent_artifact_rows,
        "dataset_totals": dataset_totals_rows,
    }
    text_report = build_text_report(
        md_source_match_rows,
        commit_distribution_rows,
        commit_agent_rows,
        collection_summary_rows,
        cross_source_term_rows,
        top_terms_rows,
        classification_rows,
        md_breakdown_row,
        pr_creator_comment_rows,
        comment_agent_count_rows,
        agent_artifact_rows,
        dataset_totals_rows,
    )
    wordcloud_svg = build_wordcloud_svg(top_terms_rows)

    write_json(OUTPUT_DIR / "clone_terms_summary.json", summary_payload)
    (OUTPUT_DIR / "clone_terms_summary.txt").write_text(text_report + "\n", encoding="utf-8")
    (OUTPUT_DIR / "clone_terms_wordcloud.svg").write_text(wordcloud_svg + "\n", encoding="utf-8")
    print(text_report)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")


def main() -> None:
    md_analysis = load_json(MD_ANALYSIS_PATH)
    md_report = load_json(MD_REPORT_PATH)
    commits_analysis = load_json(COMMITS_ANALYSIS_PATH)
    commits_report = load_json(COMMITS_REPORT_PATH)
    prs_results = load_json(PRS_RESULTS_PATH)
    prs_checkpoint = load_json(PRS_CHECKPOINT_PATH)

    md_term_counts, md_display_variants = count_terms_in_md(md_report)
    commit_term_counts, commit_display_variants = count_terms_in_commits(commits_report)
    pr_term_counts, pr_display_variants = count_terms_in_prs(prs_results)
    pr_comment_term_counts, pr_comment_display_variants = count_terms_in_pr_comments(prs_results)
    display_variants = md_display_variants

    for normalized_term, variants in commit_display_variants.items():
        display_variants.setdefault(normalized_term, Counter()).update(variants)

    for normalized_term, variants in pr_display_variants.items():
        display_variants.setdefault(normalized_term, Counter()).update(variants)

    for normalized_term, variants in pr_comment_display_variants.items():
        display_variants.setdefault(normalized_term, Counter()).update(variants)

    top_terms_rows = build_top_terms_rows(
        md_term_counts,
        commit_term_counts,
        pr_term_counts,
        display_variants,
    )
    cross_source_term_rows = build_cross_source_term_rows(
        md_term_counts,
        commit_term_counts,
        pr_comment_term_counts,
        display_variants,
    )
    md_source_match_rows = build_md_source_match_rows(md_analysis)
    commit_distribution_rows = build_commit_distribution_rows(commits_analysis)
    commit_agent_rows = build_commit_agent_rows(commits_report)
    collection_summary_rows = build_collection_summary_rows(
        md_report,
        commits_report,
        prs_results,
        prs_checkpoint,
    )
    classification_rows = build_classification_rows(
        md_analysis,
        commits_analysis,
        prs_results,
    )
    md_breakdown_row = build_md_breakdown_row(md_analysis)
    pr_creator_comment_rows = build_pr_creator_comment_rows(prs_results)
    comment_agent_count_rows = build_comment_agent_count_rows(prs_results)
    agent_artifact_rows = build_agent_artifact_rows(
        md_report,
        commits_analysis,
        comment_agent_count_rows,
    )
    dataset_totals_rows = build_dataset_totals_rows(
        md_analysis,
        md_report,
        commits_analysis,
        commits_report,
        prs_results,
        prs_checkpoint,
    )

    validate_md_total(md_analysis, md_breakdown_row)
    write_outputs(
        md_source_match_rows,
        commit_distribution_rows,
        commit_agent_rows,
        collection_summary_rows,
        cross_source_term_rows,
        top_terms_rows,
        classification_rows,
        md_breakdown_row,
        pr_creator_comment_rows,
        comment_agent_count_rows,
        agent_artifact_rows,
        dataset_totals_rows,
    )


if __name__ == "__main__":
    main()
