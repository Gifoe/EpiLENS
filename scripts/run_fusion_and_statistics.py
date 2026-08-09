#!/usr/bin/env python3
"""Run CDEL fusion, permutation, paired bootstrap, and boundary analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epilens.analyses import (
    align_branches,
    audit_primary_oof_ledger,
    boundary_channel_analysis,
    evaluate_cdel_by_fold,
    fusion_weight_sensitivity,
    paired_patient_bootstrap,
    within_patient_bcr_permutation,
)
from epilens.evaluation import patient_metrics, select_threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prq-ledger", required=True)
    parser.add_argument("--bcr-ledger", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--bootstraps", type=int, default=10000)
    args = parser.parse_args()
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    prq, bcr = pd.read_csv(args.prq_ledger), pd.read_csv(args.bcr_ledger)
    audit_primary_oof_ledger(prq)
    audit_primary_oof_ledger(bcr)
    aligned = align_branches(prq, bcr)
    cdel_test, thresholds = evaluate_cdel_by_fold(aligned)
    prq_thresholds, bcr_thresholds = [], []
    for (seed, fold), group in prq.query("partition == 'validation'").groupby(["seed", "outer_fold"]):
        prq_thresholds.append({"seed": seed, "outer_fold": fold, "threshold": select_threshold(group).threshold})
    bcr_for_threshold = bcr.copy()
    bcr_for_threshold["probability_nez"] = 1 / (1 + np.exp(bcr_for_threshold.logit_ez))
    for (seed, fold), group in bcr_for_threshold.query("partition == 'validation'").groupby(["seed", "outer_fold"]):
        bcr_thresholds.append({"seed": seed, "outer_fold": fold, "threshold": select_threshold(group).threshold})
    prq_thresholds, bcr_thresholds = pd.DataFrame(prq_thresholds), pd.DataFrame(bcr_thresholds)
    prq_thresholds.to_csv(output / "prq_validation_thresholds.csv", index=False)
    bcr_thresholds.to_csv(output / "bcr_validation_thresholds.csv", index=False)
    cdel_test.to_csv(output / "cdel_test_channel_ledger.csv", index=False)
    thresholds.to_csv(output / "cdel_validation_thresholds.csv", index=False)
    fusion_weight_sensitivity(aligned).to_csv(
        output / "fusion_weight_sensitivity.csv", index=False
    )
    permutation = within_patient_bcr_permutation(aligned, thresholds, args.permutations)
    permutation.to_csv(
        output / "within_patient_permutation.csv", index=False
    )
    patient_rows = []
    for model, ledger, threshold_table in (
        ("PRQ-Net", prq.loc[prq.partition == "test"], None),
        ("BCR-Net", bcr_for_threshold.loc[bcr_for_threshold.partition == "test"], None),
        ("CDEL", cdel_test, thresholds),
    ):
        for (seed, fold), group in ledger.groupby(["seed", "outer_fold"]):
            if "selected_threshold" in group:
                threshold = float(group.selected_threshold.iloc[0])
            elif threshold_table is not None:
                threshold = float(threshold_table.query("seed == @seed and outer_fold == @fold").threshold.iloc[0])
            else:
                validation = (prq if model == "PRQ-Net" else bcr_for_threshold).query("seed == @seed and outer_fold == @fold and partition == 'validation'").copy()
                threshold = select_threshold(validation).threshold
            table = patient_metrics(group, threshold)
            table["model"], table["seed"], table["outer_fold"] = model, seed, fold
            patient_rows.append(table)
    patients = pd.concat(patient_rows, ignore_index=True)
    patients.to_csv(output / "patient_metrics.csv", index=False)
    comparison = paired_patient_bootstrap(patients, "CDEL", "PRQ-Net", "macro_f1", args.bootstraps)
    (output / "cdel_vs_prq_bootstrap.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    observed = float(patients.query("model == 'CDEL'").macro_f1.mean())
    permutation_summary = {
        "observed_patient_macro_f1": observed,
        "null_mean": float(permutation.patient_macro_f1.mean()),
        "one_sided_corrected_p": float(
            (1 + (permutation.patient_macro_f1 >= observed).sum()) / (1 + len(permutation))
        ),
    }
    (output / "within_patient_permutation_summary.json").write_text(
        json.dumps(permutation_summary, indent=2), encoding="utf-8"
    )
    boundary_channel_analysis(
        prq.query("partition == 'test'"),
        bcr_for_threshold.query("partition == 'test'"),
        cdel_test,
        prq_thresholds,
        bcr_thresholds,
    ).to_csv(output / "boundary_channel_analysis.csv", index=False)


if __name__ == "__main__":
    main()
