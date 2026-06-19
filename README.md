# Dataset Code Duplication Discussions

This repository contains scripts and generated results for mining code-duplication and code-reuse discussions from the AI-Config dataset and related GitHub sources.

The project focuses on three complementary signals:

- Context and skill files listed in the AI-Config dataset.
- Matching context and skill files discovered directly via GitHub Code Search.
- Commit messages and pull requests that mention duplication-related terms.

## Repository Structure

### Input data

- `ai_config/`: CSV exports from the AI-Config dataset.
- `ai_config/repos_data/repos/`: repository-level data used by the broader dataset pipeline.

### Scripts

- `1_get_context_files.py`: scans AI-Config CSV sources for relevant Markdown files and detects duplication-related terms.
- `2_get_context_files_github.py`: uses GitHub Code Search plus repository quality filters to find context and skill files discussing duplication or reuse topics.
- `3_get_commits.py`: analyzes commit messages for duplication-related terms.
- `4_get_prs.py`: searches pull requests and comments for duplication-related discussions.
- `6_get_issues.py`: searches closed issues and comments for duplication-related discussions.

### Output directories

- `1_context_results/`: report and analysis for AI-Config context-file mining.
- `2_context_github_results/`: report, analysis, and checkpoint data for GitHub-discovered context and skill files.
- `3_commits_results/`: report and analysis for commit-message mining.
- `4_prs_results/`: report and checkpoint data for pull-request mining.
- `6_issues_results/`: report and checkpoint data for issue mining.
- `ai_config_results/`, `context_skill_heading_results/`, `exclusive_results/`: additional generated analyses.

## Requirements

- Python 3.10+
- A virtual environment is recommended.
- GitHub API token(s) in `.env` or the environment:
  - `GITHUB_TOKEN`
  - or a rotating pool such as `GITHUB_TOKEN_1` to `GITHUB_TOKEN_9`

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

Run the scripts from the repository root.

```bash
python3 1_get_context_files.py
python3 2_get_context_files_github.py
python3 3_get_commits.py
python3 4_get_prs.py
python3 6_get_issues.py
```

## What Each Script Produces

### `1_get_context_files.py`

Generates:

- `1_context_results/clone_terms_report.json`
- `1_context_results/clone_terms_analysis.json`
- `1_context_results/clone_terms_failed.json`

### `2_get_context_files_github.py`

Generates:

- `2_context_github_results/clone_terms_report.json`
- `2_context_github_results/clone_terms_analysis.json`
- `2_context_github_results/clone_terms_checkpoint.json`

This script now supports resumable execution through a checkpoint file that persists:

- discovered candidate files,
- repository quality-check outcomes,
- per-file processing outcomes.

### `3_get_commits.py`

Generates:

- `3_commits_results/commits_clone_terms_report.json`
- `3_commits_results/commits_clone_terms_analysis.json`

### `4_get_prs.py`

Generates:

- `4_prs_results/search_prs_results.json`
- `4_prs_results/search_prs_checkpoint.json`

### `6_get_issues.py`

Generates:

- `6_issues_results/search_issues_results.json`
- `6_issues_results/search_issues_checkpoint.json`

## Detection Scope

The scripts search for English duplication-related expressions such as:

- duplicate code
- code duplication
- repeated code
- DRY principle
- code reuse
- code sharing
- don't repeat yourself
- redundant code

Depending on the script, matches are classified using headings, body mentions, commit text, or pull-request discussion context.

## Notes

- The repository includes generated result files, so reruns may update tracked JSON outputs.
- GitHub-driven scripts are rate-limit sensitive and may take substantial time depending on the available token pool.
- `2_get_context_files_github.py`, `4_get_prs.py`, and `6_get_issues.py` are designed for long-running collection tasks.

## Related Papers

| Title | Notes |
|---|---|
| **Configuring Agentic AI Coding Tools: An Exploratory Study** | Primary paper by Sebastian Baltes et al. and the direct source of the AI-Config dataset used here. Zenodo: [10.5281/zenodo.19696190](https://zenodo.org/records/19696190) · arXiv: [2510.21413](https://arxiv.org/abs/2510.21413) |
| **Context Engineering for AI Agents in Open-Source Software** | A related study by Sebastian Baltes, presented at the **MSR** (Mining Software Repositories) conference. Independent from the AI-Config dataset but part of the same broader research line on agentic AI coding tools. |
