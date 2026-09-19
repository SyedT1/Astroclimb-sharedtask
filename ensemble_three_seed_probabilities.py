#!/usr/bin/env python3
"""Average AstroCLIMB seed probabilities and create a validated submission."""

import csv
import math
from pathlib import Path


TARGETS = ["same_figure", "same_paper", "related_papers", "unrelated_papers"]
PROBABILITY_COLUMNS = [f"p_{target}" for target in TARGETS]
SEEDS = (17, 42, 123)
ROOT = Path(__file__).resolve().parent
INPUTS = {
    seed: ROOT
    / f"astroclimb_qwen3vl8b_language_seed{seed}_qlora"
    / "submission_probabilities.csv"
    for seed in SEEDS
}
SAMPLE_SUBMISSION = ROOT / "data" / "sample_submission.csv"
OUTPUT_DIR = ROOT / "astroclimb_qwen3vl8b_three_seed_ensemble"


def read_probabilities(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["id", *PROBABILITY_COLUMNS]
        if reader.fieldnames != expected:
            raise ValueError(f"{path}: expected columns {expected}, got {reader.fieldnames}")
        rows: dict[str, list[float]] = {}
        for row in reader:
            row_id = str(row["id"])
            if row_id in rows:
                raise ValueError(f"{path}: duplicate id {row_id}")
            values = [float(row[column]) for column in PROBABILITY_COLUMNS]
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise ValueError(f"{path}: invalid probabilities for id {row_id}: {values}")
            if not math.isclose(sum(values), 1.0, abs_tol=1e-5):
                raise ValueError(f"{path}: probabilities do not sum to one for id {row_id}")
            rows[row_id] = values
    return rows


def read_submission_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = ["id", *TARGETS]
        if (reader.fieldnames or [])[: len(required)] != required:
            raise ValueError(f"{path}: unexpected columns {reader.fieldnames}")
        ids = [str(row["id"]) for row in reader]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate IDs")
    return ids


def main() -> None:
    components = {seed: read_probabilities(path) for seed, path in INPUTS.items()}
    ordered_ids = read_submission_ids(SAMPLE_SUBMISSION)
    expected_ids = set(ordered_ids)
    for seed, rows in components.items():
        if set(rows) != expected_ids:
            missing = len(expected_ids - set(rows))
            extra = len(set(rows) - expected_ids)
            raise ValueError(f"Seed {seed}: ID mismatch, missing={missing}, extra={extra}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    probability_path = OUTPUT_DIR / "submission_probabilities.csv"
    submission_path = OUTPUT_DIR / "submission.csv"
    prediction_counts = [0] * len(TARGETS)

    with (
        probability_path.open("w", encoding="utf-8", newline="") as probability_handle,
        submission_path.open("w", encoding="utf-8", newline="") as submission_handle,
    ):
        probability_writer = csv.writer(probability_handle, lineterminator="\n")
        submission_writer = csv.writer(submission_handle, lineterminator="\n")
        probability_writer.writerow(["id", *PROBABILITY_COLUMNS])
        submission_writer.writerow(["id", *TARGETS])

        for row_id in ordered_ids:
            averaged = [
                sum(components[seed][row_id][class_index] for seed in SEEDS) / len(SEEDS)
                for class_index in range(len(TARGETS))
            ]
            total = sum(averaged)
            averaged = [value / total for value in averaged]
            prediction = max(range(len(TARGETS)), key=averaged.__getitem__)
            one_hot = [int(index == prediction) for index in range(len(TARGETS))]
            prediction_counts[prediction] += 1
            probability_writer.writerow([row_id, *[f"{value:.16g}" for value in averaged]])
            submission_writer.writerow([row_id, *one_hot])

    if len(ordered_ids) != 10_000:
        raise ValueError(f"Expected 10,000 rows, got {len(ordered_ids)}")
    print(f"Wrote {probability_path}")
    print(f"Wrote {submission_path}")
    print("Prediction counts:", dict(zip(TARGETS, prediction_counts)))


if __name__ == "__main__":
    main()
