import numpy as np

from epilens.data import PatientRecord
from epilens.features import feature_baseline_vector, fit_standardizer, four_view_expansion


def _record(patient_id: str, offset: float = 0.0) -> PatientRecord:
    descriptors = np.arange(2 * 3 * 4 * 9, dtype=np.float32).reshape(2, 3, 4, 9)
    descriptors = descriptors / 100 + offset
    return PatientRecord(
        patient_id=patient_id,
        center="private_a",
        descriptors=descriptors,
        window_times=np.array([[-2, -1, 1], [-2, -1, 1]], dtype=np.float32),
        valid=np.ones((2, 3, 4), dtype=bool),
        channel_names=("A", "B", "C", "D"),
        label_nez=np.array([0, 1, 1, 1]),
    )


def test_four_views_use_only_pre_onset_reference():
    record = _record("patient_01")
    expanded, valid = four_view_expansion(
        record.descriptors, record.window_times, record.valid
    )
    expected = record.descriptors[0, :2].mean(axis=0)
    np.testing.assert_allclose(expanded[0, 2, :, 9:18], record.descriptors[0, 2] - expected)
    assert expanded.shape == (2, 3, 4, 36)
    assert valid.all()


def test_fit_only_standardizer_and_baseline_dimension():
    fit = _record("patient_01")
    held_out = _record("patient_02", offset=1000)
    standardizer = fit_standardizer([fit])
    assert np.max(standardizer.mean) < 10
    assert feature_baseline_vector(held_out).shape == (4, 88)
