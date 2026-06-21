from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_REPORT_PATH = ROOT_DIR / "1_configurations_artifacts_results" / "clone_terms_report.json"
COMMITS_REPORT_PATH = ROOT_DIR / "2_commits_results" / "commits_clone_terms_report.json"
OUTPUT_DIR = ROOT_DIR / "3_sample_results"
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / "stratified_sample.csv"
DEFAULT_OUTPUT_METADATA = OUTPUT_DIR / "stratified_sample_metadata.json"
CONFIDENCE_LEVEL = 0.95
CONFIDENCE_Z_SCORE = 1.96
MARGIN_OF_ERROR = 0.05
RESPONSE_DISTRIBUTION = 0.5
MIN_PER_STRATUM = 1
RANDOM_SEED = 42


@dataclass(frozen=True)
class CandidateRecord:
	dataset: str
	artifact_type: str
	source_group: str
	repo_name: str
	item_id: str
	file_name: str
	file_path: str
	branch: str
	timestamp: str
	url: str
	matched_terms: str
	matched_term_count: int
	total_occurrences: int
	stratum: str

def load_json(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as file_handle:
		return json.load(file_handle)


def write_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as file_handle:
		json.dump(payload, file_handle, indent=2, ensure_ascii=False)
		file_handle.write("\n")


def normalize_whitespace(value: str) -> str:
	return " ".join(value.split())


def collect_terms(term_entries: list[dict[str, Any]]) -> tuple[str, int, int]:
	terms: list[str] = []
	total_occurrences = 0

	for entry in term_entries:
		term = str(entry.get("term") or "").strip()
		if term:
			terms.append(term)

		count = int(entry.get("count") or 0)
		total_occurrences += count

	unique_terms = sorted(set(terms), key=str.lower)
	return "; ".join(unique_terms), len(unique_terms), total_occurrences


def build_stratum(dataset: str, artifact_type: str) -> str:
	return f"{dataset}|{artifact_type}"


def flatten_configuration_matches(report_data: dict[str, Any]) -> list[CandidateRecord]:
	records: list[CandidateRecord] = []

	for source in report_data.get("sources", []):
		csv_name = str(source.get("csv") or "unknown.csv")
		artifact_type = csv_name.removesuffix(".csv") or "unknown"

		for match in source.get("matches", []):
			matched_terms, matched_term_count, total_occurrences = collect_terms(match.get("terms", []))

			records.append(
				CandidateRecord(
					dataset="configuration_artifact",
					artifact_type=artifact_type,
					source_group=csv_name,
					repo_name=str(match.get("repo_name") or ""),
					item_id=str(match.get("raw_url") or match.get("github_link") or match.get("file_path") or ""),
					file_name=str(match.get("file_name") or ""),
					file_path=str(match.get("file_path") or ""),
					branch=str(match.get("branch") or ""),
					timestamp=str(match.get("timestamp") or ""),
					url=str(match.get("github_link") or match.get("raw_url") or ""),
					matched_terms=matched_terms,
					matched_term_count=matched_term_count,
					total_occurrences=int(match.get("total_occurrences") or total_occurrences),
					stratum=build_stratum("configuration_artifact", artifact_type),
				)
			)

	return records


def flatten_commit_matches(report_data: dict[str, Any]) -> list[CandidateRecord]:
	records: list[CandidateRecord] = []

	for match in report_data.get("matches", []):
		matched_terms, matched_term_count, total_occurrences = collect_terms(match.get("terms", []))

		records.append(
			CandidateRecord(
				dataset="commit",
				artifact_type="commit",
				source_group=str(match.get("ai_tool") or "unknown"),
				repo_name=str(match.get("repo_name") or ""),
				item_id=str(match.get("commit_sha") or match.get("github_link") or ""),
				file_name="",
				file_path="",
				branch=str(match.get("branch") or ""),
				timestamp=str(match.get("commit_timestamp") or ""),
				url=str(match.get("github_link") or ""),
				matched_terms=matched_terms,
				matched_term_count=matched_term_count,
				total_occurrences=int(match.get("total_occurrences") or total_occurrences),
				stratum=build_stratum("commit", "commit"),
			)
		)

	return records


def build_population() -> list[CandidateRecord]:
	config_report = load_json(CONFIG_REPORT_PATH)
	commits_report = load_json(COMMITS_REPORT_PATH)
	return [
		*flatten_configuration_matches(config_report),
		*flatten_commit_matches(commits_report),
	]


def group_by_stratum(records: list[CandidateRecord]) -> dict[str, list[CandidateRecord]]:
	grouped: dict[str, list[CandidateRecord]] = defaultdict(list)
	for record in records:
		grouped[record.stratum].append(record)
	return dict(grouped)


def calculate_sample_size(population_size: int) -> int:
	if population_size <= 0:
		raise ValueError("population_size must be greater than zero.")

	variance = RESPONSE_DISTRIBUTION * (1 - RESPONSE_DISTRIBUTION)
	infinite_population_size = (CONFIDENCE_Z_SCORE ** 2 * variance) / (MARGIN_OF_ERROR ** 2)
	adjusted_sample_size = infinite_population_size / (
		1 + ((infinite_population_size - 1) / population_size)
	)
	return min(population_size, max(1, round(adjusted_sample_size)))


def allocate_sample_sizes(
	grouped_records: dict[str, list[CandidateRecord]],
	requested_sample_size: int,
	min_per_stratum: int,
) -> dict[str, int]:
	if requested_sample_size <= 0:
		raise ValueError("sample_size must be greater than zero.")

	total_population = sum(len(records) for records in grouped_records.values())
	if total_population == 0:
		raise ValueError("No candidate records were found in the input reports.")

	sample_size = min(requested_sample_size, total_population)
	strata = sorted(grouped_records)
	allocations = {stratum: 0 for stratum in strata}

	if sample_size == total_population:
		return {stratum: len(grouped_records[stratum]) for stratum in strata}

	min_targets = {
		stratum: min(len(grouped_records[stratum]), max(min_per_stratum, 0))
		for stratum in strata
	}

	if sum(min_targets.values()) > sample_size:
		ranked_strata = sorted(strata, key=lambda item: (-len(grouped_records[item]), item))
		for stratum in ranked_strata[:sample_size]:
			allocations[stratum] = 1
		return allocations

	allocations.update(min_targets)
	remaining = sample_size - sum(allocations.values())
	if remaining == 0:
		return allocations

	remaining_capacity = {
		stratum: len(grouped_records[stratum]) - allocations[stratum]
		for stratum in strata
	}
	total_remaining_capacity = sum(remaining_capacity.values())

	fractional_parts: list[tuple[float, str]] = []
	for stratum in strata:
		capacity = remaining_capacity[stratum]
		if capacity <= 0:
			continue

		exact = remaining * (capacity / total_remaining_capacity)
		extra = min(int(exact), capacity)
		allocations[stratum] += extra
		fractional_parts.append((exact - extra, stratum))

	assigned = sum(allocations.values())
	leftover = sample_size - assigned
	if leftover <= 0:
		return allocations

	for _, stratum in sorted(fractional_parts, key=lambda item: (-item[0], item[1])):
		if leftover == 0:
			break
		if allocations[stratum] < len(grouped_records[stratum]):
			allocations[stratum] += 1
			leftover -= 1

	return allocations


def draw_stratified_sample(
	grouped_records: dict[str, list[CandidateRecord]],
	allocations: dict[str, int],
	seed: int,
) -> list[CandidateRecord]:
	rng = random.Random(seed)
	sampled_records: list[CandidateRecord] = []

	for stratum in sorted(grouped_records):
		records = list(grouped_records[stratum])
		target_size = allocations.get(stratum, 0)
		if target_size <= 0:
			continue

		if target_size >= len(records):
			chosen = records
		else:
			chosen = rng.sample(records, target_size)

		sampled_records.extend(
			sorted(chosen, key=lambda record: (record.repo_name.lower(), record.item_id.lower()))
		)

	return sampled_records


def write_sample_csv(path: Path, sampled_records: list[CandidateRecord]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"sample_id",
		"dataset",
		"artifact_type",
		"source_group",
		"stratum",
		"repo_name",
		"item_id",
		"file_name",
		"file_path",
		"branch",
		"timestamp",
		"url",
		"matched_terms",
		"matched_term_count",
		"total_occurrences",
		"True Code Duplication?",
	]

	with path.open("w", encoding="utf-8", newline="") as file_handle:
		writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
		writer.writeheader()

		for index, record in enumerate(sampled_records, start=1):
			row = asdict(record)
			row["sample_id"] = index
			row["True Code Duplication?"] = ""
			writer.writerow({field: row.get(field, "") for field in fieldnames})


def build_metadata(
	population: list[CandidateRecord],
	sampled_records: list[CandidateRecord],
	allocations: dict[str, int],
	target_sample_size: int,
) -> dict[str, Any]:
	grouped_population = group_by_stratum(population)
	grouped_sample = group_by_stratum(sampled_records)

	return {
		"target_sample_size": target_sample_size,
		"actual_sample_size": len(sampled_records),
		"population_size": len(population),
		"confidence_level": CONFIDENCE_LEVEL,
		"confidence_z_score": CONFIDENCE_Z_SCORE,
		"margin_of_error": MARGIN_OF_ERROR,
		"response_distribution": RESPONSE_DISTRIBUTION,
		"seed": RANDOM_SEED,
		"min_per_stratum": MIN_PER_STRATUM,
		"strata": [
			{
				"stratum": stratum,
				"population": len(grouped_population[stratum]),
				"sampled": len(grouped_sample.get(stratum, [])),
				"allocated": allocations.get(stratum, 0),
			}
			for stratum in sorted(grouped_population)
		],
	}


def main() -> None:
	population = build_population()
	target_sample_size = calculate_sample_size(len(population))
	grouped_records = group_by_stratum(population)
	allocations = allocate_sample_sizes(grouped_records, target_sample_size, MIN_PER_STRATUM)
	sampled_records = draw_stratified_sample(grouped_records, allocations, RANDOM_SEED)

	write_sample_csv(DEFAULT_OUTPUT_CSV, sampled_records)

	metadata = build_metadata(
		population=population,
		sampled_records=sampled_records,
		allocations=allocations,
		target_sample_size=target_sample_size,
	)
	write_json(DEFAULT_OUTPUT_METADATA, metadata)

	print(f"Population size: {len(population)}")
	print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")
	print(f"Margin of error: {MARGIN_OF_ERROR:.0%}")
	print(f"Sample size: {len(sampled_records)}")
	print(f"CSV written to: {DEFAULT_OUTPUT_CSV}")
	print(f"Metadata written to: {DEFAULT_OUTPUT_METADATA}")


if __name__ == "__main__":
	main()
