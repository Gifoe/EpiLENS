#!/usr/bin/env python3
"""Run the strongest reported baseline: logistic regression + patient z-score."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epilens.baseline import (
    build_logistic_patient_z_estimator,
    patient_feature_zscore,
)
from epilens.data import load_partition_manifest, load_records, validate_protocol
from epilens.evaluation import select_threshold
from epilens.features import feature_baseline_vector


def _channel_table(records) -> pd.DataFrame:
    rows = []
    for record in records:
        values = feature_baseline_vector(record)
        for index, channel in enumerate(record.channel_names):
            row = {
                "patient_id": record.patient_id,
                "center": record.center,
                "channel_name": channel,
                "label_nez": int(record.label_nez[index]),
            }
            row.update(
                {
                    f"feature_{column:02d}": value
                    for column, value in enumerate(values[index])
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--expected-patients", type=int, default=80)
    args = parser.parse_args()

    records = load_records(args.records)
    manifest = load_partition_manifest(args.partition_manifest)
    validate_protocol(records, manifest, expected_patients=args.expected_patients)
    table = _channel_table(records)
    feature_columns = [column for column in table if column.startswith("feature_")]
    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    ledgers = []
    seeds = [int(value) for value in args.seeds.split(",")]
    for seed in seeds:
        for fold in sorted(manifest["outer_fold"].unique()):
            frames = {}
            for role in ("fit", "validation", "test"):
                patient_ids = set(
                    manifest.loc[
                        (manifest["outer_fold"] == fold)
                        & (manifest["partition"] == role),
                        "patient_id",
                    ]
                )
                frame = table.loc[table["patient_id"].isin(patient_ids)].copy()
                frames[role] = patient_feature_zscore(frame, feature_columns)

            estimator = build_logistic_patient_z_estimator(seed)
            estimator.fit(
                frames["fit"][feature_columns], frames["fit"]["label_nez"]
            )
            validation = frames["validation"].copy()
            test = frames["test"].copy()
            validation["probability_nez"] = estimator.predict_proba(
                validation[feature_columns]
            )[:, 1]
            test["probability_nez"] = estimator.predict_proba(
                test[feature_columns]
            )[:, 1]
            test["selected_threshold"] = select_threshold(validation).threshold
            test["outer_fold"] = fold
            test["seed"] = seed
            test["model"] = "Logistic_Patient_Z"
            ledgers.append(test)

    pd.concat(ledgers, ignore_index=True).to_csv(
        output_directory / "Logistic_Patient_Z_channel_ledger.csv", index=False
    )


if __name__ == "__main__":
    main()
