"""Strongest comparison model reported with EpiLENS.

The retained baseline is logistic regression applied to label-free,
within-patient z-scored channel features. The paper identifies this method as
the strongest baseline under the primary patient-macro Macro-F1 metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def patient_feature_zscore(
    frame: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Apply label-free feature normalization independently within patients."""
    result = frame.copy()
    for _, indices in result.groupby("patient_id").groups.items():
        values = result.loc[indices, feature_columns].to_numpy(dtype=float)
        mean = np.nanmean(values, axis=0)
        scale = np.nanstd(values, axis=0)
        scale = np.where(scale > 1e-6, scale, 1.0)
        result.loc[indices, feature_columns] = (values - mean) / scale
    return result


def build_logistic_patient_z_estimator(seed: int) -> Pipeline:
    """Return the fixed logistic-regression configuration used in the paper."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    solver="liblinear",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
