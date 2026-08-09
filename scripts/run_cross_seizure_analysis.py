#!/usr/bin/env python3
"""Evaluate frozen PRQ-Net/BCR-Net checkpoints with one, two, or all seizures."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from epilens.data import load_partition_manifest, load_records
from epilens.analyses import paired_condition_bootstrap
from epilens.evaluation import patient_metrics
from epilens.training import load_trained_branch, predict_records


def _subset(record, indices):
    return replace(
        record,
        descriptors=record.descriptors[indices],
        window_times=record.window_times[indices],
        valid=record.valid[indices],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--partition-manifest", required=True)
    parser.add_argument("--prq-checkpoints", required=True)
    parser.add_argument("--bcr-checkpoints", required=True)
    parser.add_argument("--prq-thresholds", required=True)
    parser.add_argument("--bcr-thresholds", required=True)
    parser.add_argument("--cdel-thresholds", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    records = {record.patient_id: record for record in load_records(args.records)}
    manifest = load_partition_manifest(args.partition_manifest)
    threshold_maps = {}
    for model, path in (
        ("PRQ-Net", args.prq_thresholds),
        ("BCR-Net", args.bcr_thresholds),
        ("CDEL", args.cdel_thresholds),
    ):
        table = pd.read_csv(path)
        threshold_maps[model] = {
            (int(row.seed), int(row.outer_fold)): float(row.threshold)
            for row in table.itertuples()
        }
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outputs = []
    for seed in map(int, args.seeds.split(",")):
        for fold in sorted(manifest.outer_fold.unique()):
            ids = manifest.query("outer_fold == @fold and partition == 'test'").patient_id
            branches = {}
            for branch, directory in (("prq", args.prq_checkpoints), ("bcr", args.bcr_checkpoints)):
                checkpoint = Path(directory) / f"{branch}_seed_{seed}_fold_{fold}.pt"
                branches[branch] = load_trained_branch(checkpoint, device)
            for patient_id in ids:
                record = records[patient_id]
                seizure_total = record.descriptors.shape[0]
                if seizure_total < 2:
                    continue
                random = np.random.default_rng(2027 + seed + fold + sum(map(ord, patient_id)))
                conditions = [("all", 0, np.arange(seizure_total))]
                for count in (1, 2):
                    conditions.extend(
                        (str(count), repeat, np.sort(random.choice(seizure_total, count, replace=False)))
                        for repeat in range(args.repeats)
                    )
                for count, repeat, indices in conditions:
                    selected = _subset(record, indices)
                    ledgers = {}
                    for branch, (model, standardizer, _) in branches.items():
                        ledgers[branch] = predict_records(model, [selected], standardizer, branch, device)
                    joined = ledgers["prq"].merge(
                        ledgers["bcr"][["patient_id", "channel_name", "logit_ez"]],
                        on=["patient_id", "channel_name"],
                        validate="one_to_one",
                    )
                    joined["probability_nez"] = 0.8 * joined["probability_nez"] + 0.2 / (1 + np.exp(joined["logit_ez"]))
                    model_ledgers = {
                        "PRQ-Net": ledgers["prq"],
                        "BCR-Net": ledgers["bcr"].assign(
                            probability_nez=lambda x: 1 / (1 + np.exp(x.logit_ez))
                        ),
                        "CDEL": joined,
                    }
                    for model_name, frame in model_ledgers.items():
                        frame = frame.copy()
                        frame["model"] = model_name
                        frame["seizure_count"], frame["repeat"] = str(count), repeat
                        frame["seed"], frame["outer_fold"] = seed, fold
                        outputs.append(frame)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ledger = pd.concat(outputs, ignore_index=True)
    ledger.to_csv(destination, index=False)
    metric_rows = []
    for (model, seed, fold, count, repeat), group in ledger.groupby(
        ["model", "seed", "outer_fold", "seizure_count", "repeat"]
    ):
        validation_threshold = threshold_maps[model][(int(seed), int(fold))]
        patient = patient_metrics(group, validation_threshold)
        patient["seed"], patient["outer_fold"] = seed, fold
        patient["model"] = model
        patient["seizure_count"], patient["repeat"] = str(count), repeat
        metric_rows.append(patient)
    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(destination.with_name(destination.stem + "_patient_metrics.csv"), index=False)
    comparisons = {}
    cdel_metrics = metrics.query("model == 'CDEL'")
    for reference in ("1", "2"):
        for metric in ("macro_f1", "ez_f1", "ez_auprc", "ndcg_ez"):
            comparisons[f"all_vs_{reference}_{metric}"] = paired_condition_bootstrap(
                cdel_metrics, "all", reference, metric, args.bootstraps
            )
    destination.with_name(destination.stem + "_bootstrap.json").write_text(
        json.dumps(comparisons, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
