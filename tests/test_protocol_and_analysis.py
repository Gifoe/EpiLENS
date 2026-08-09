import numpy as np
import pandas as pd
import pytest

from epilens.analyses import add_cdel_probability, align_branches
from epilens.data import PatientRecord, validate_protocol
from epilens.evaluation import select_threshold


def _records(count=5):
    return [
        PatientRecord(
            patient_id=f"patient_{index:02d}",
            center="private_a",
            descriptors=np.zeros((1, 2, 2, 9), dtype=np.float32),
            window_times=np.array([[-1, 1]], dtype=np.float32),
            valid=np.ones((1, 2, 2), dtype=bool),
            channel_names=("A", "B"),
            label_nez=np.array([0, 1]),
        )
        for index in range(count)
    ]


def test_protocol_rejects_outer_test_duplication():
    records = _records()
    rows = []
    for fold in range(5):
        for index, record in enumerate(records):
            rows.append(
                {
                    "patient_id": record.patient_id,
                    "outer_fold": fold,
                    "partition": "test" if index == fold else "fit",
                }
            )
    manifest = pd.DataFrame(rows)
    validate_protocol(records, manifest, expected_patients=5)
    manifest.loc[manifest.index[-2], "partition"] = "test"
    with pytest.raises(ValueError):
        validate_protocol(records, manifest, expected_patients=5)


def test_branch_alignment_and_locked_fusion():
    keys = {
        "seed": 42,
        "outer_fold": 0,
        "patient_id": "patient_00",
        "center": "private_a",
        "channel_name": "A",
        "label_nez": 1,
        "partition": "test",
    }
    prq = pd.DataFrame([{**keys, "probability_nez": 0.75}])
    bcr = pd.DataFrame([{**keys, "logit_ez": 0.0}])
    fused = add_cdel_probability(align_branches(prq, bcr))
    assert fused.probability_nez.iloc[0] == pytest.approx(0.7)


def test_threshold_grid_and_tie_break_are_deterministic():
    ledger = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "B"],
            "label_nez": [0, 1, 0, 1],
            "probability_nez": [0.1, 0.9, 0.2, 0.8],
        }
    )
    selected = select_threshold(ledger)
    assert selected.threshold / 0.005 == pytest.approx(
        round(selected.threshold / 0.005)
    )
    assert selected.patient_macro_f1 == pytest.approx(1)
