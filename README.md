# Dataset Code Duplication Discussions

This repository contains scripts and generated results for mining code-duplication and code-reuse discussions from the AI-Config dataset and related GitHub sources.

The project focuses on three complementary signals:

- Context and skill files listed in the AI-Config dataset.
- Commit messages and pull requests that mention duplication-related terms.

## Repository Structure

### Input data

- `ai_config/`: CSV exports from the AI-Config dataset.
- `ai_config/repos_data/repos/`: repository-level data used by the broader dataset pipeline.

### Scripts

- `1_get_configurations_artifacts.py`: scans AI-Config CSV sources for relevant Markdown files and detects duplication-related terms.
- `2_get_commits.py`: analyzes commit messages for duplication-related terms.
- `3_get_prs.py`: searches pull requests and comments for duplication-related discussions.
- `4_analyze.py`: consolidates the markdown, commit, and pull-request outputs into summary tables and visualizations.

### Output directories

- `1_configurations_artifacts_results/`: report and analysis for AI-Config context-file mining.
- `2_commits_results/`: report and analysis for commit-message mining.
- `3_prs_results/`: report and checkpoint data for pull-request mining.
- `4_analyze_results/`: consolidated summaries and visualizations.
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
python3 1_get_configurations_artifacts.py
python3 2_get_commits.py
python3 3_get_prs.py
python3 4_analyze.py
```

## What Each Script Produces

### `1_get_configurations_artifacts.py`

Generates:

- `1_configurations_artifacts_results/clone_terms_report.json`
- `1_configurations_artifacts_results/clone_terms_analysis.json`
- `1_configurations_artifacts_results/clone_terms_failed.json`

### `2_get_commits.py`

Generates:

- `2_commits_results/commits_clone_terms_report.json`
- `2_commits_results/commits_clone_terms_analysis.json`

### `3_get_prs.py`

Generates:

- `3_prs_results/search_prs_results.json`
- `3_prs_results/search_prs_checkpoint.json`

### `7_get_true_configuration_artifact_prs.py`

Generates:

- `7_true_configuration_artifact_prs_results/merged_prs_after_true_configuration_artifacts.json`
- `7_true_configuration_artifact_prs_results/merged_prs_after_true_configuration_artifacts_checkpoint.json`

### `9_get_true_configuration_artifact_agent_commits.py`

Generates:

- `9_true_configuration_artifact_agent_commits_results/agent_commits_after_true_configuration_artifacts.json`

### `4_analyze.py`

Generates:

- `4_analyze_results/clone_terms_summary.json`
- `4_analyze_results/clone_terms_summary.txt`
- `4_analyze_results/clone_terms_wordcloud.svg`

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
- `3_get_prs.py` is designed for a long-running collection task.

## Related Papers

| Title | Notes |
|---|---|
| **Configuring Agentic AI Coding Tools: An Exploratory Study** | Primary paper by Sebastian Baltes et al. and the direct source of the AI-Config dataset used here. Zenodo: [10.5281/zenodo.19696190](https://zenodo.org/records/19696190) · arXiv: [2510.21413](https://arxiv.org/abs/2510.21413) |
| **Context Engineering for AI Agents in Open-Source Software** | A related study by Sebastian Baltes, presented at the **MSR** (Mining Software Repositories) conference. Independent from the AI-Config dataset but part of the same broader research line on agentic AI coding tools. |
