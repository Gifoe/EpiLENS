#!/usr/bin/env python3
"""Retrain the PRQ-Net and BCR-Net variants reported in Table 3."""

from __future__ import annotations

import argparse
from pathlib import Path

from epilens.data import load_partition_manifest, load_records, validate_protocol
from epilens.training import TrainingConfig, train_outer_fold


PRQ_VARIANTS = {
    "Base": dict(quantile=0.10, use_temporal=False, use_patient_relative=True),
    "Temporal": dict(
        quantile=0.10, use_temporal=True, use_patient_relative=True,
        temporal_aggregation="mean",
    ),
    "Q10_Full": dict(
        quantile=0.10, use_temporal=True, use_patient_relative=True,
        temporal_aggregation="quantile",
    ),
    "Without_Patient_Relative": dict(
        quantile=0.10, use_temporal=True, use_patient_relative=False
    ),
}
BCR_VARIANTS = {
    "BCE": dict(boundary_weight=0.0, coverage_weight=0.0),
    "Q10_Control": dict(
        boundary_weight=0.0, coverage_weight=0.0, quantile_control=True
    ),
    "Boundary": dict(boundary_weight=0.05, coverage_weight=0.0),
    "Coverage": dict(boundary_weight=0.0, coverage_weight=0.08),
    "Boundary_Coverage": dict(boundary_weight=0.05, coverage_weight=0.08),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    records = load_records(args.records)
    manifest = load_partition_manifest(args.partition_manifest)
    validate_protocol(records, manifest)
    config = TrainingConfig(device=args.device)
    for seed in [int(value) for value in args.seeds.split(",")]:
        for fold in sorted(manifest["outer_fold"].unique()):
            for name, settings in PRQ_VARIANTS.items():
                train_outer_fold(
                    "prq",
                    records,
                    manifest,
                    int(fold),
                    seed,
                    Path(args.output_directory) / "PRQ-Net" / name,
                    config,
                    **settings,
                )
            for name, settings in BCR_VARIANTS.items():
                train_outer_fold(
                    "bcr",
                    records,
                    manifest,
                    int(fold),
                    seed,
                    Path(args.output_directory) / "BCR-Net" / name,
                    config,
                    **settings,
                )


if __name__ == "__main__":
    main()
