import numpy as np
import pandas as pd

from epilens.baseline import (
    build_logistic_patient_z_estimator,
    patient_feature_zscore,
)


def test_patient_feature_zscore_is_patient_local():
    frame = pd.DataFrame(
        {
            "patient_id": ["a", "a", "b", "b"],
            "feature_00": [1.0, 3.0, 10.0, 14.0],
            "feature_01": [5.0, 5.0, 2.0, 6.0],
        }
    )
    result = patient_feature_zscore(frame, ["feature_00", "feature_01"])
    grouped = result.groupby("patient_id")[["feature_00", "feature_01"]]
    np.testing.assert_allclose(grouped.mean().to_numpy(), 0.0, atol=1e-12)
    assert np.isfinite(result[["feature_00", "feature_01"]]).all().all()


def test_best_baseline_returns_probabilities():
    estimator = build_logistic_patient_z_estimator(seed=42)
    features = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    labels = np.array([0, 0, 1, 1])
    estimator.fit(features, labels)
    probabilities = estimator.predict_proba(features)
    assert probabilities.shape == (4, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
