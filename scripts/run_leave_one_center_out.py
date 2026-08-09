#!/usr/bin/env python3
"""Train both branches under an explicit leave-one-center-out manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epilens.data import load_partition_manifest, load_records
from epilens.training import TrainingConfig, train_outer_fold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--loco-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    records = load_records(args.records)
    manifest = load_partition_manifest(args.loco_manifest)
    center_by_patient = {record.patient_id: record.center for record in records}
    for fold, group in manifest.groupby("outer_fold"):
        held_out = {center_by_patient[patient] for patient in group.query("partition == 'test'").patient_id}
        if len(held_out) != 1:
            raise ValueError(f"LOCO fold {fold} must test exactly one center")
        training_centers = {center_by_patient[patient] for patient in group.query("partition != 'test'").patient_id}
        if held_out & training_centers:
            raise ValueError(f"Held-out center leaks into training in fold {fold}")
    config = TrainingConfig(device=args.device)
    output = Path(args.output_directory)
    for seed in map(int, args.seeds.split(",")):
        for fold in sorted(manifest.outer_fold.unique()):
            for branch in ("prq", "bcr"):
                train_outer_fold(branch, records, manifest, int(fold), seed, output / branch, config)


if __name__ == "__main__":
    main()
