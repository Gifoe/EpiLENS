#!/usr/bin/env python3
"""Train PRQ-Net on the fixed five-fold, three-seed protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from epilens.data import load_partition_manifest, load_records, validate_protocol
from epilens.training import TrainingConfig, train_outer_fold


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
            train_outer_fold(
                "prq",
                records,
                manifest,
                int(fold),
                seed,
                Path(args.output_directory) / "PRQ-Net",
                config,
                quantile=0.10,
                use_temporal=True,
                use_patient_relative=True,
            )


if __name__ == "__main__":
    main()
