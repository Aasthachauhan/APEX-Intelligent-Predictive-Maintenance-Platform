"""Row-wise (stateless) feature engineering -> no train/test leakage by construction.
temp_diff, power (torque*omega), strain (wear*torque) mirror quantities in the AI4I docs.
CAVEAT: AI4I labels are rule-derived from these quantities, so high scores are expected
and say little about real-world difficulty."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml.data.ai4i import FEATURES_CAT, FEATURES_NUM

ENGINEERED = ["temp_diff", "power_w", "strain", "wear_rate"]


class AI4IFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["temp_diff"] = X["process_temp"] - X["air_temp"]
        X["power_w"] = X["torque"] * X["rpm"] * 2 * np.pi / 60
        X["strain"] = X["tool_wear"] * X["torque"]
        X["wear_rate"] = X["tool_wear"] / X["rpm"].clip(lower=1)
        return X


def feature_names():
    return FEATURES_NUM + ENGINEERED + ["type_H", "type_L", "type_M"]


def build_preprocessor() -> Pipeline:
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), FEATURES_NUM + ENGINEERED),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ]
    )
    return Pipeline([("fe", AI4IFeatures()), ("ct", ct)])
