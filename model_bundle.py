"""Scaler + classifier bundle used by training and inference."""
from __future__ import annotations

import pandas as pd


class DiseaseModel:
    """Accepts raw feature DataFrames / dicts and applies saved scaling."""

    def __init__(self, model, scaler, numeric_cols, feature_order, threshold=0.5):
        self.model = model
        self.scaler = scaler
        self.numeric_cols = list(numeric_cols)
        self.feature_order = list(feature_order)
        self.threshold = float(threshold)

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(X).copy()
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_order].apply(pd.to_numeric, errors="coerce").fillna(0)
        if self.numeric_cols and self.scaler is not None:
            df[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        return df

    def predict(self, X):
        df = self._prepare(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[:, 1]
            return (proba >= self.threshold).astype(int)
        return self.model.predict(df)

    def predict_proba(self, X):
        df = self._prepare(X)
        return self.model.predict_proba(df)
