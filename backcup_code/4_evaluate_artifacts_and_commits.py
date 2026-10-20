from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


ROOT_DIR = Path(__file__).resolve().parent
CONFIGURATION_REPORT_PATH = ROOT_DIR / "1_configurations_artifacts_results" / "clone_terms_report.json"
COMMITS_REPORT_PATH = ROOT_DIR / "2_commits_results" / "commits_clone_terms_report.json"
OUTPUT_DIR = ROOT_DIR / "4_artifacts_and_commits_review_results"
RESULTS_PATH = OUTPUT_DIR / "gpt_5_mini_artifacts_and_commits_review.json"
CHECKPOINT_PATH = OUTPUT_DIR / "gpt_5_mini_artifacts_and_commits_checkpoint.json"
LOCAL_REPOS_DIR = ROOT_DIR / "ai_config" / "repos_data" / "repos"

MODEL_NAME = "gpt-5-mini"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_TEMPERATURE: float | None = None
FETCH_TIMEOUT_SECONDS = 30
MODEL_TIMEOUT_SECONDS = 180
MODEL_MAX_CHARS = 20000
REQUEST_DELAY_SECONDS = 0.0
CHECKPOINT_SAVE_EVERY = 5
RESULTS_SAVE_EVERY = 25
MAX_CONFIGURATION_ARTIFACTS: int | None = None
MAX_COMMITS: int | None = None

SYSTEM_PROMPT_ARTIFACTS = """You are an expert software engineering researcher analyzing whether a configuration artifact provides real evidence related to code duplication.
Your task is to classify whether the provided Markdown configuration artifact is genuinely related to code duplication.

Classification instructions:
First, classify the artifact as:

* "true": the artifact provides real evidence related to code duplication.
* "false": the artifact does not provide real evidence related to code duplication.

Then, classify the evidence as:

* "dedicated": the evidence appears in the main Markdown title, such as a `#` heading, or in the introductory description directly associated with that main title. An artifact can also be classified as "dedicated" even without a matching title, if the title contains only refactoring terms but the artifact's overall objective is to remove duplicate code.
* "partial": the evidence appears in a Markdown subtitle, such as `##`, `###`, `####`, or another lower-level heading.
* "mention": the evidence appears only in descriptive text, bullet points, examples, notes, or implementation details.
* "none": the match is a false positive and does not refer to duplicated source code or repeated implementation logic.

Caution points:
When evaluating as "true", consider that:

* Common refactoring terms can indicate positive evidence of duplication reduction when combined with code duplication terms.
* The code duplication terms used as matched terms are generally reliable indicators of real duplication-related evidence.

When evaluating as "false", consider that:

* The matched term may refer to sharing code as a repository, rather than reusing source code.
* The content may refer to duplicated configuration files, not duplicated source code.
* The content may refer to duplicated documentation files, not duplicated source code.
* Be careful with "avoid duplication," as it may refer to other matters that are not code duplicates.

Response instructions:
Return only valid JSON, using the following schema:

{
  "is_real": true,
  "classification": "dedicated | partial | mention | none",
  "matched_term": "the matched term from the input",
  "reason": "Briefly explain the classification using only the provided evidence."
}

Return JSON only. Do not wrap the JSON in markdown fences."""

SYSTEM_PROMPT_COMMITS = """You are an expert software engineering researcher analyzing whether an agent commit provides real evidence related to code duplication.
Your task is to classify whether the provided commit is genuinely related to code duplication.

Classification instructions:
Classify the commit as:
* "true": the commit provides real evidence related to code duplication.
* "false": the commit does not provide real evidence related to code duplication.

Classify the commit as:
* "dedicated": the main commit title contains evidence related to code duplication, or the whole commit is mainly focused on reducing or refactoring duplicated code. A commit can also be classified as "dedicated" even without a matching title, if the title contains only refactoring terms but the commit's overall objective is to remove duplicate code.
* "partial": the commit appears to be a squash commit with multiple internal commit titles, and one internal title refers to duplicate code, code reuse, redundant logic, or refactoring to reduce duplication.
* "mention": the duplication evidence appears only in a descriptive text or bullet point, without being the main title or an internal commit title.
* "none": the match is a false positive and does not refer to duplicated source code or repeated implementation logic.

Caution points:
When evaluating as "true", consider that:

* Common refactoring terms can indicate positive evidence of duplication reduction, when combined with code duplication terms.
* The code duplication terms used as matched terms are generally reliable indicators of real duplication-related evidence.

When evaluating as "false", consider that:

* The matched term refers to sharing code as a repository, rather than reusing source code.
* The content refers to duplicated configuration files, not duplicated source code.
* The content refers to duplicated documentation files, not duplicated source code.
* Be careful with "avoid duplication," as it may refer to other matters that are not code duplicates.

Response instructions:
Return only valid JSON, using the following schema:

{
  "is_real": true,
  "classification": "dedicated | partial | mention | none",
  "matched_term": "the matched term from the input",
  "reason": "Briefly explain the classification using only the provided evidence."
}

Return JSON only. Do not wrap the JSON in markdown fences."""


@dataclass(frozen=True)
class ReviewTarget:
    source_kind: str
    artifact_type: str
    source_group: str
    repo_name: str
    item_id: str
    url: str
    raw_url: str
    branch: str
    timestamp: str
    file_name: str
    file_path: str
    commit_sha: str
    ai_tool: str
    commit_message: str
    matched_term: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    with temp_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")
    temp_path.replace(path)


def flatten_configuration_report(report_data: dict[str, Any]) -> list[ReviewTarget]:
    records: list[ReviewTarget] = []

    for source in report_data.get("sources", []):
        csv_name = str(source.get("csv") or "unknown.csv")
        artifact_type = csv_name.removesuffix(".csv") or "unknown"

        for match in source.get("matches", []):
            raw_url = str(match.get("raw_url") or "")
            github_link = str(match.get("github_link") or "")
            item_id = raw_url or github_link or str(match.get("file_path") or "")
            terms = match.get("terms") or []
            matched_term = str(terms[0].get("term") or "") if terms else ""

            records.append(
                ReviewTarget(
                    source_kind="configuration_artifact",
                    artifact_type=artifact_type,
                    source_group=csv_name,
                    repo_name=str(match.get("repo_name") or ""),
                    item_id=item_id,
                    url=github_link,
                    raw_url=raw_url,
                    branch=str(match.get("branch") or ""),
                    timestamp=str(match.get("timestamp") or ""),
                    file_name=str(match.get("file_name") or ""),
                    file_path=str(match.get("file_path") or ""),
                    commit_sha="",
                    ai_tool="",
                    commit_message="",
                    matched_term=matched_term,
                )
            )

    return records


def build_commit_item_id(repo_name: str, commit_sha: str, github_link: str) -> str:
    repo_name = repo_name.strip()
    commit_sha = commit_sha.strip()
    github_link = github_link.strip()
    if repo_name and commit_sha:
        return f"{repo_name}::{commit_sha}"
    return commit_sha or github_link


def flatten_commits_report(report_data: dict[str, Any]) -> list[ReviewTarget]:
    records: list[ReviewTarget] = []

    for match in report_data.get("matches", []):
        repo_name = str(match.get("repo_name") or "")
        github_link = str(match.get("github_link") or "")
        commit_sha = str(match.get("commit_sha") or "")
        item_id = build_commit_item_id(repo_name, commit_sha, github_link)
        terms = match.get("terms") or []
        matched_term = str(terms[0].get("term") or "") if terms else ""

        records.append(
            ReviewTarget(
                source_kind="commit",
                artifact_type="commit",
                source_group="commits_clone_terms_report.json",
                repo_name=repo_name,
                item_id=item_id,
                url=github_link,
                raw_url="",
                branch=str(match.get("branch") or ""),
                timestamp=str(match.get("commit_timestamp") or ""),
                file_name="",
                file_path="",
                commit_sha=commit_sha,
                ai_tool=str(match.get("ai_tool") or ""),
                commit_message=str(match.get("commit_message") or ""),
                matched_term=matched_term,
            )
        )

    return records


def build_headers() -> dict[str, str]:
    api_key = os.getenv(OPENAI_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing {OPENAI_API_KEY_ENV} in the environment or .env file.")

    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def fetch_artifact_content(raw_url: str) -> str:
    response = requests.get(raw_url, timeout=FETCH_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.text


def read_text_with_fallbacks(path: Path) -> str:
    raw_bytes = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def build_local_artifact_path(record: ReviewTarget) -> Path:
    repo_dir_name = record.repo_name.replace("/", "§")
    return LOCAL_REPOS_DIR / repo_dir_name / record.file_path


def read_local_artifact_content(record: ReviewTarget) -> str:
    local_path = build_local_artifact_path(record)
    return read_text_with_fallbacks(local_path)


def trim_for_model(text: str, max_chars: int = MODEL_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}\n\n[Truncated for model input]"



def _extract_title(record: ReviewTarget, content: str) -> str:
    if record.source_kind == "configuration_artifact":
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        return record.file_name or record.file_path
    return content.split("\n", 1)[0].strip()


def build_user_prompt(record: ReviewTarget, content: str) -> str:
    file_type = record.artifact_type if record.source_kind == "configuration_artifact" else "commit"
    title = _extract_title(record, content)
    content_label = "Artifact content" if record.source_kind == "configuration_artifact" else "Commit content"

    return (
        f"File type:\n{file_type}\n\n"
        f"Matched term:\n{record.matched_term}\n\n"
        f"Title:\n{title}\n\n"
        f"{content_label}:\n{trim_for_model(content)}"
    )


def extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_chunks: list[str] = []
    for output_item in payload.get("output", []) or []:
        for content_item in output_item.get("content", []) or []:
            text = content_item.get("text")
            if isinstance(text, str) and text.strip():
                output_chunks.append(text.strip())

    if output_chunks:
        return "\n".join(output_chunks)

    choices = payload.get("choices", []) or []
    for choice in choices:
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    raise ValueError("Could not extract model text from the API response.")


def parse_model_json(response_text: str) -> dict[str, Any]:
    candidate = response_text.strip()

    if candidate.startswith("```"):
        lines = candidate.splitlines()
        if len(lines) >= 3:
            candidate = "\n".join(lines[1:-1]).strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end != -1 and end >= start:
        candidate = candidate[start : end + 1]

    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("The model response is not a JSON object.")
    return parsed


def evaluate_target(record: ReviewTarget, content: str, headers: dict[str, str], system_prompt: str) -> tuple[dict[str, Any], str]:
    body: dict[str, Any] = {
        "model": MODEL_NAME,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": build_user_prompt(record, content)}],
            },
        ],
    }
    if MODEL_TEMPERATURE is not None:
        body["temperature"] = MODEL_TEMPERATURE

    response = requests.post(
        f"{OPENAI_BASE_URL.rstrip('/')}/responses",
        headers=headers,
        json=body,
        timeout=MODEL_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    response_text = extract_response_text(payload)
    return parse_model_json(response_text), response_text


def get_target_content(record: ReviewTarget) -> str:
    if record.source_kind == "configuration_artifact":
        local_path = build_local_artifact_path(record)
        if local_path.exists():
            return read_local_artifact_content(record)

        if not record.raw_url:
            raise ValueError(
                f"Missing local artifact and raw_url for configuration artifact: {local_path}"
            )
        return fetch_artifact_content(record.raw_url)

    if not record.commit_message:
        raise ValueError("Missing commit_message in commit record.")
    return record.commit_message


def load_checkpoint() -> dict[str, dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return {}

    payload = load_json(CHECKPOINT_PATH)
    processed: dict[str, dict[str, Any]] = {}
    for result in payload.get("results", []):
        source_kind = str(result.get("source_kind") or "")
        if source_kind == "commit":
            item_id = build_commit_item_id(
                str(result.get("repo_name") or ""),
                str(result.get("commit_sha") or ""),
                str(result.get("url") or result.get("github_link") or ""),
            )
            result["item_id"] = item_id
        else:
            item_id = str(result.get("item_id") or "")
        if item_id:
            processed[item_id] = result
    return processed


def save_checkpoint(results: list[dict[str, Any]], total_targets: int) -> None:
    write_json(
        CHECKPOINT_PATH,
        {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_targets": total_targets,
            "processed_targets": len(results),
            "results": results,
        },
    )


def build_decision_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(result.get("classification") or result.get("status") for result in results)
    return dict(sorted(counts.items()))


def build_kind_counts(results: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(result.get("source_kind") or "unknown" for result in results)
    return dict(sorted(counts.items()))


def save_results(results: list[dict[str, Any]], total_targets: int) -> None:
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "prompts": {
            "artifacts": SYSTEM_PROMPT_ARTIFACTS,
            "commits": SYSTEM_PROMPT_COMMITS,
        },
        "total_targets": total_targets,
        "processed_targets": len(results),
        "source_kind_counts": build_kind_counts(results),
        "decision_counts": build_decision_counts(results),
        "results": results,
    }
    write_json(RESULTS_PATH, output)


def persist_progress(
    processed_by_id: dict[str, dict[str, Any]],
    total_targets: int,
    *,
    save_review_file: bool,
) -> None:
    current_results = list(processed_by_id.values())
    save_checkpoint(current_results, total_targets)
    if save_review_file:
        save_results(current_results, total_targets)


def print_startup_report(configuration_records: list[ReviewTarget], commit_records: list[ReviewTarget]) -> None:
    configuration_counts = Counter(record.artifact_type for record in configuration_records)

    print("Evaluation order:")
    print(f"  1. Configuration artifacts: {len(configuration_records)}")
    for artifact_type, count in sorted(configuration_counts.items()):
        print(f"     - {artifact_type}: {count}")
    print(f"  2. Commits: {len(commit_records)}")


def evaluate_records(
    records: list[ReviewTarget],
    processed_by_id: dict[str, dict[str, Any]],
    headers: dict[str, str],
    total_targets: int,
    start_index: int,
    system_prompt: str,
) -> int:
    current_index = start_index

    for record in records:
        current_index += 1
        label = record.file_path if record.source_kind == "configuration_artifact" else record.commit_sha

        if record.item_id in processed_by_id:
            print(f"[{current_index}/{total_targets}] SKIP already processed: {record.repo_name} / {label}")
            continue

        print(f"[{current_index}/{total_targets}] Evaluating {record.source_kind}: {record.repo_name} / {label}")
        result: dict[str, Any] = {
            **asdict(record),
            "status": "pending",
        }

        try:
            content = get_target_content(record)
            evaluation, response_text = evaluate_target(record, content, headers, system_prompt)
            result.update(
                {
                    "status": "evaluated",
                    "is_real": evaluation.get("is_real"),
                    "classification": evaluation.get("classification"),
                    "matched_term_response": evaluation.get("matched_term"),
                    "reason": evaluation.get("reason"),
                    "model_response_text": response_text,
                    "content_length": len(content),
                }
            )
        except Exception as exc:
            result.update(
                {
                    "status": "error",
                    "is_real": None,
                    "classification": None,
                    "matched_term_response": None,
                    "reason": str(exc),
                    "model_response_text": None,
                    "content_length": None,
                }
            )

        processed_by_id[record.item_id] = result
        processed_count = len(processed_by_id)
        should_save_checkpoint = (
            result["status"] == "error"
            or processed_count % CHECKPOINT_SAVE_EVERY == 0
            or processed_count == total_targets
        )
        should_save_results = (
            result["status"] == "error"
            or processed_count % RESULTS_SAVE_EVERY == 0
            or processed_count == total_targets
        )
        if should_save_checkpoint or should_save_results:
            persist_progress(
                processed_by_id,
                total_targets,
                save_review_file=should_save_results,
            )

        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    return current_index


def main() -> None:
    configuration_report_data = load_json(CONFIGURATION_REPORT_PATH)
    commits_report_data = load_json(COMMITS_REPORT_PATH)

    configuration_records = flatten_configuration_report(configuration_report_data)
    commit_records = flatten_commits_report(commits_report_data)

    if MAX_CONFIGURATION_ARTIFACTS is not None:
        configuration_records = configuration_records[:MAX_CONFIGURATION_ARTIFACTS]
    if MAX_COMMITS is not None:
        commit_records = commit_records[:MAX_COMMITS]

    print_startup_report(configuration_records, commit_records)

    headers = build_headers()
    processed_by_id = load_checkpoint()
    total_targets = len(configuration_records) + len(commit_records)

    current_index = 0
    current_index = evaluate_records(
        records=configuration_records,
        processed_by_id=processed_by_id,
        headers=headers,
        total_targets=total_targets,
        start_index=current_index,
        system_prompt=SYSTEM_PROMPT_ARTIFACTS,
    )
    current_index = evaluate_records(
        records=commit_records,
        processed_by_id=processed_by_id,
        headers=headers,
        total_targets=total_targets,
        start_index=current_index,
        system_prompt=SYSTEM_PROMPT_COMMITS,
    )

    persist_progress(processed_by_id, total_targets, save_review_file=True)
    print(f"Processed targets: {current_index}")
    print(f"Results written to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
