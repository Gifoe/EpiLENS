#!/usr/bin/env python3
"""Align independently trained branches and evaluate locked CDEL fusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from epilens.analyses import (
    align_branches,
    audit_primary_oof_ledger,
    evaluate_cdel_by_fold,
)
from epilens.evaluation import patient_metrics, summarize_patient_equal


def _read_ledgers(directory: str | Path) -> pd.DataFrame:
    paths = sorted(Path(directory).glob("*_fold_*_validation.csv"))
    paths += sorted(Path(directory).glob("*_fold_*_test.csv"))
    if not paths:
        raise FileNotFoundError(f"No branch ledgers in {directory}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prq-directory", required=True)
    parser.add_argument("--bcr-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    prq = _read_ledgers(args.prq_directory)
    bcr = _read_ledgers(args.bcr_directory)
    audit_primary_oof_ledger(prq)
    audit_primary_oof_ledger(bcr)
    aligned = align_branches(prq, bcr)
    test, thresholds = evaluate_cdel_by_fold(aligned)
    patient_parts = []
    summary_rows = []
    for (seed, fold), group in test.groupby(["seed", "outer_fold"]):
        threshold = float(group["selected_threshold"].iloc[0])
        patient = patient_metrics(group, threshold)
        patient["seed"] = seed
        patient["outer_fold"] = fold
        patient["model"] = "CDEL"
        patient_parts.append(patient)
        summary_rows.append(
            {
                "seed": seed,
                "outer_fold": fold,
                **summarize_patient_equal(group, threshold),
            }
        )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    test.to_csv(output / "CDEL_channel_ledger.csv", index=False)
    thresholds.to_csv(output / "CDEL_validation_thresholds.csv", index=False)
    pd.concat(patient_parts, ignore_index=True).to_csv(
        output / "CDEL_patient_metrics.csv", index=False
    )
    pd.DataFrame(summary_rows).to_csv(output / "CDEL_fold_metrics.csv", index=False)


if __name__ == "__main__":
    main()
