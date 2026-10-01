"""
Time-Series Feature Engineering for Energy Consumption Forecasting.
Generates cyclical encodings, autoregressive lags, rolling statistics,
and handles multi-step horizon feature construction.
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


class TimeSeriesFeatureEngineer:
    """
    Transforms time series data into ML-ready tabular feature matrix
    with no lookahead data leakage.
    """

    def __init__(
        self,
        target_col: str = "energy_consumption_mw",
        freq: str = "1h",
        lags: Optional[List[int]] = None,
        rolling_windows: Optional[List[int]] = None
    ):
        self.target_col = target_col
        self.freq = freq.lower()

        # Sensible defaults based on frequency
        if lags is None:
            if "h" in self.freq:
                self.lags = [1, 2, 3, 24, 48, 168]
            else:
                self.lags = [1, 2, 3, 7, 14, 30]
        else:
            self.lags = lags

        if rolling_windows is None:
            if "h" in self.freq:
                self.rolling_windows = [6, 24, 168]
            else:
                self.rolling_windows = [3, 7, 30]
        else:
            self.rolling_windows = rolling_windows

        self.feature_names: List[str] = []

    def create_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Creates temporal, lag, rolling, and interaction features.
        Expects a DataFrame with a DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Input DataFrame must have a DatetimeIndex.")

        df_feat = df.copy()

        # 1. Cyclical Calendar Features
        idx = df_feat.index
        hours = idx.hour.values
        dayofweek = idx.dayofweek.values
        months = idx.month.values
        dayofyear = idx.dayofyear.values

        # Hour of day (24h period)
        df_feat["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
        df_feat["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)

        # Day of week (7d period)
        df_feat["dayofweek_sin"] = np.sin(2 * np.pi * dayofweek / 7.0)
        df_feat["dayofweek_cos"] = np.cos(2 * np.pi * dayofweek / 7.0)

        # Month of year (12m period)
        df_feat["month_sin"] = np.sin(2 * np.pi * (months - 1) / 12.0)
        df_feat["month_cos"] = np.cos(2 * np.pi * (months - 1) / 12.0)

        # Day of year (Annual seasonality)
        df_feat["dayofyear_sin"] = np.sin(2 * np.pi * (dayofyear - 1) / 365.25)
        df_feat["dayofyear_cos"] = np.cos(2 * np.pi * (dayofyear - 1) / 365.25)

        # Calendar indicators
        df_feat["is_weekend"] = (dayofweek >= 5).astype(int)
        df_feat["is_peak_hour"] = (((hours >= 8) & (hours <= 11)) | ((hours >= 17) & (hours <= 21))).astype(int)
        df_feat["quarter"] = idx.quarter.values

        # 2. Exogenous temperature features (if present)
        for temp_col in ["temperature_c", "temperature", "temp"]:
            if temp_col in df_feat.columns:
                t_val = df_feat[temp_col]
                df_feat["cooling_degree"] = np.maximum(0.0, t_val - 22.0)
                df_feat["heating_degree"] = np.maximum(0.0, 16.0 - t_val)
                break

        # 3. Autoregressive Lags (Shifted to ensure no lookahead)
        if self.target_col not in df_feat.columns:
            num_cols = df_feat.select_dtypes(include=[np.number]).columns.tolist()
            if num_cols:
                self.target_col = num_cols[0]
            else:
                raise KeyError(f"Target column '{self.target_col}' not found in DataFrame. Available: {list(df_feat.columns)}")

        target_series = df_feat[self.target_col]
        for lag in self.lags:
            col_name = f"lag_{lag}"
            df_feat[col_name] = target_series.shift(lag)

        # 4. Rolling Statistics (Shifted by 1 so target_t is NEVER in the rolling window)
        shifted_target = target_series.shift(1)
        for window in self.rolling_windows:
            df_feat[f"rolling_mean_{window}"] = shifted_target.rolling(window=window).mean()
            df_feat[f"rolling_std_{window}"] = shifted_target.rolling(window=window).std().fillna(0.0)
            df_feat[f"rolling_min_{window}"] = shifted_target.rolling(window=window).min()
            df_feat[f"rolling_max_{window}"] = shifted_target.rolling(window=window).max()

        # 5. Drop NaN rows resulting from lagging if in training mode
        if is_training:
            df_feat = df_feat.dropna()

        # Feature column identification
        exclude_cols = [self.target_col, "region", "zone", "area", "timestamp", "datetime"]
        self.feature_names = [c for c in df_feat.columns if c not in exclude_cols]

        return df_feat

    def get_feature_matrix(self, df_feat: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Splits feature dataframe into X (features) and y (target).
        """
        if self.target_col not in df_feat.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")
        X = df_feat[self.feature_names]
        y = df_feat[self.target_col]
        return X, y
