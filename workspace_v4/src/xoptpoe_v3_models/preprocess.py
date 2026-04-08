"""Missingness-aware preprocessing for XOPTPOE v3 neural models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FittedPreprocessor:
    """Train-only fitted preprocessing state."""

    feature_names: list[str]
    original_feature_names: list[str]
    indicator_feature_names: list[str]
    fill_values: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        work = frame.copy()
        original = work[self.original_feature_names].apply(pd.to_numeric, errors="coerce")
        indicators: dict[str, pd.Series] = {}
        for feature_name in self.indicator_feature_names:
            indicators[f"{feature_name}__missing"] = original[feature_name].isna().astype(float)

        imputed = original.copy()
        for feature_name, fill_value in self.fill_values.items():
            imputed[feature_name] = imputed[feature_name].fillna(fill_value)

        scaled = imputed.copy()
        for feature_name in self.original_feature_names:
            mean = self.means[feature_name]
            std = self.stds[feature_name]
            scaled[feature_name] = (scaled[feature_name] - mean) / std

        transformed = scaled.copy()
        for name, series in indicators.items():
            transformed[name] = series

        transformed = transformed[self.feature_names]
        return transformed.to_numpy(dtype=np.float32), transformed



def fit_preprocessor(
    train_df: pd.DataFrame,
    *,
    feature_manifest: pd.DataFrame,
    feature_columns: list[str],
) -> FittedPreprocessor:
    """Fit train-only imputation and scaling for a selected feature set."""
    manifest = feature_manifest.loc[feature_manifest["feature_name"].isin(feature_columns)].copy()
    manifest = manifest.set_index("feature_name")

    original = train_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    fill_values: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    indicator_features: list[str] = []

    for feature_name in feature_columns:
        if feature_name not in manifest.index:
            raise ValueError(f"Feature '{feature_name}' missing from feature manifest")
        row = manifest.loc[feature_name]
        strategy = str(row["imputation_strategy_hint"])
        series = original[feature_name]

        if int(row["missing_indicator_recommended"]) == 1:
            indicator_features.append(feature_name)

        if strategy == "zero_fill_with_indicator":
            fill_value = 0.0
        else:
            fill_value = float(series.median()) if series.notna().any() else 0.0
        fill_values[feature_name] = fill_value

        filled = series.fillna(fill_value)
        mean = float(filled.mean())
        std = float(filled.std(ddof=1)) if len(filled) > 1 else 1.0
        if not np.isfinite(std) or std <= 1e-8:
            std = 1.0
        means[feature_name] = mean
        stds[feature_name] = std

    indicator_feature_names = [f for f in feature_columns if f in indicator_features]
    output_feature_names = list(feature_columns) + [f"{name}__missing" for name in indicator_feature_names]

    return FittedPreprocessor(
        feature_names=output_feature_names,
        original_feature_names=list(feature_columns),
        indicator_feature_names=indicator_feature_names,
        fill_values=fill_values,
        means=means,
        stds=stds,
    )
