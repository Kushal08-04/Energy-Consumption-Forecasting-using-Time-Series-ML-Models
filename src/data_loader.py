"""
Data Loader & Preprocessor for Energy Consumption Forecasting.
Supports multi-region ingestion, automated column detection, resampling, outlier handling,
and generation of realistic multi-region benchmark datasets.
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd


def generate_benchmark_energy_data(
    start_date: str = "2023-01-01",
    end_date: str = "2026-03-31",
    freq: str = "1h",
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic multi-region hourly energy consumption dataset
    with realistic diurnal, weekly, seasonal cycles, weather effects, and concept drift.
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(date_range)

    regions_data = []

    # Configs for regions
    region_configs = {
        "North-Metro": {
            "base_load": 3200.0,
            "daily_amp": 900.0,
            "weekend_factor": 0.82,
            "summer_peak_boost": 650.0,
            "winter_peak_boost": 500.0,
            "temp_mean": 18.0,
            "temp_amp": 14.0,
            "noise_std": 60.0,
            "drift_start_idx": int(n * 0.80),
            "drift_multiplier": 1.15, # 15% structural growth/shift
        },
        "South-Coast": {
            "base_load": 2400.0,
            "daily_amp": 750.0,
            "weekend_factor": 0.88,
            "summer_peak_boost": 950.0, # High summer cooling
            "winter_peak_boost": 250.0,
            "temp_mean": 24.0,
            "temp_amp": 10.0,
            "noise_std": 50.0,
            "drift_start_idx": int(n * 0.82),
            "drift_multiplier": 1.12,
        },
        "Central-Industrial": {
            "base_load": 4100.0,
            "daily_amp": 1200.0,
            "weekend_factor": 0.70, # Heavy drop on weekends
            "summer_peak_boost": 400.0,
            "winter_peak_boost": 450.0,
            "temp_mean": 16.0,
            "temp_amp": 15.0,
            "noise_std": 80.0,
            "drift_start_idx": int(n * 0.85),
            "drift_multiplier": 1.10,
        },
    }

    hours = date_range.hour.values
    dayofweek = date_range.dayofweek.values
    dayofyear = date_range.dayofyear.values

    # Base diurnal pattern (morning rush 8-10, evening peak 18-21, night trough 3-5)
    diurnal_curve = (
        -0.6 * np.cos(2 * np.pi * (hours - 4) / 24)
        + 0.4 * np.sin(4 * np.pi * (hours - 8) / 24)
    )

    # Seasonal pattern (summer peak ~day 200, winter peak ~day 15)
    summer_curve = np.clip(np.sin(2 * np.pi * (dayofyear - 80) / 365), 0, 1)
    winter_curve = np.clip(np.cos(2 * np.pi * (dayofyear - 15) / 365), 0, 1)

    for region_name, cfg in region_configs.items():
        # Temperature simulation
        temp_season = cfg["temp_mean"] + cfg["temp_amp"] * np.sin(2 * np.pi * (dayofyear - 100) / 365)
        temp_daily = 5.0 * np.sin(2 * np.pi * (hours - 9) / 24)
        temperature = temp_season + temp_daily + np.random.normal(0, 2.5, size=n)

        # Base load calculation
        load = cfg["base_load"] + cfg["daily_amp"] * diurnal_curve

        # Weekend attenuation
        weekend_mask = dayofweek >= 5
        load = np.where(weekend_mask, load * cfg["weekend_factor"], load)

        # Weather / Seasonal demand: U-shaped demand response (cooling degree + heating degree)
        cooling_demand = np.maximum(0, temperature - 22.0) * (cfg["summer_peak_boost"] / 12.0)
        heating_demand = np.maximum(0, 14.0 - temperature) * (cfg["winter_peak_boost"] / 12.0)
        load += (summer_curve * cfg["summer_peak_boost"] * 0.4) + cooling_demand
        load += (winter_curve * cfg["winter_peak_boost"] * 0.4) + heating_demand

        # Add Gaussian noise
        noise = np.random.normal(0, cfg["noise_std"], size=n)
        load += noise

        # Inject concept drift in the latest 15-20% of data (new industrial cluster / tariff reform)
        drift_idx = cfg["drift_start_idx"]
        drift_ramp = np.linspace(1.0, cfg["drift_multiplier"], n - drift_idx)
        load[drift_idx:] = load[drift_idx:] * drift_ramp

        # Round values
        load = np.round(np.clip(load, a_min=100.0, a_max=None), 2)
        temperature = np.round(temperature, 1)

        df_region = pd.DataFrame({
            "timestamp": date_range,
            "region": region_name,
            "energy_consumption_mw": load,
            "temperature_c": temperature,
        })
        regions_data.append(df_region)

    combined_df = pd.concat(regions_data, ignore_index=True)
    return combined_df


class EnergyDataLoader:
    """
    Flexible dataset loader and preprocessor for Energy Time Series.
    """

    DATETIME_CANDIDATES = [
        "timestamp", "datetime", "date", "time", "ts", "datum",
        "date_time", "record_time", "period"
    ]

    TARGET_CANDIDATES = [
        "energy_consumption_mw", "consumption", "load", "energy",
        "mw", "kwh", "power", "demand", "value", "target", "active_power"
    ]

    REGION_CANDIDATES = [
        "region", "area", "zone", "location", "city", "grid", "country"
    ]

    def __init__(self, target_col: Optional[str] = None, datetime_col: Optional[str] = None, region_col: Optional[str] = None):
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.region_col = region_col

    def inspect_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Infers datetime, target, and region columns from a dataframe.
        """
        cols_lower = {str(c).lower().strip(): c for c in df.columns}

        # 1. Infer datetime
        found_dt = None
        if self.datetime_col and self.datetime_col in df.columns:
            found_dt = self.datetime_col
        else:
            for cand in self.DATETIME_CANDIDATES:
                if cand in cols_lower:
                    found_dt = cols_lower[cand]
                    break
            if not found_dt:
                # Try finding any column with 'date' or 'time' in name or dtype datetime
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        found_dt = col
                        break
                    elif any(k in str(col).lower() for k in ["date", "time", "year"]):
                        found_dt = col
                        break

        # 2. Infer target
        found_target = None
        if self.target_col and self.target_col in df.columns:
            found_target = self.target_col
        else:
            for cand in self.TARGET_CANDIDATES:
                if cand in cols_lower:
                    found_target = cols_lower[cand]
                    break
            if not found_target:
                # Pick first numeric column that is not the datetime column
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                for c in numeric_cols:
                    if c != found_dt:
                        found_target = c
                        break

        # 3. Infer region
        found_region = None
        if self.region_col and self.region_col in df.columns:
            found_region = self.region_col
        else:
            for cand in self.REGION_CANDIDATES:
                if cand in cols_lower:
                    found_region = cols_lower[cand]
                    break

        return {
            "datetime_col": found_dt,
            "target_col": found_target,
            "region_col": found_region,
            "all_columns": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
        }

    def process(
        self,
        df: pd.DataFrame,
        selected_region: Optional[str] = None,
        resample_freq: str = "1h",
        clean_outliers: bool = True,
        clip_iqr_multiplier: float = 3.0
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Processes and standardizes raw dataset into an aligned time-series.
        Returns:
            processed_df: DataFrame indexed by Timestamp with columns [target, ...exogenous]
            metadata: Summary information of the preprocessing run
        """
        inspection = self.inspect_columns(df)
        dt_col = self.datetime_col or inspection["datetime_col"]
        target_col = self.target_col or inspection["target_col"]
        region_col = self.region_col or inspection["region_col"]

        if not dt_col or dt_col not in df.columns:
            raise ValueError(f"Could not identify a valid datetime column. Available columns: {list(df.columns)}")
        if not target_col or target_col not in df.columns:
            raise ValueError(f"Could not identify a valid energy target column. Available columns: {list(df.columns)}")

        df_work = df.copy()

        # Filter by region if present
        available_regions = []
        if region_col and region_col in df_work.columns:
            available_regions = [str(r) for r in df_work[region_col].dropna().unique().tolist()]
            if selected_region and selected_region in available_regions:
                df_work = df_work[df_work[region_col].astype(str) == selected_region]

        # Parse datetime
        df_work[dt_col] = pd.to_datetime(df_work[dt_col], errors="coerce")
        df_work = df_work.dropna(subset=[dt_col])
        df_work = df_work.sort_values(dt_col)

        # Set datetime as index and remove duplicates
        df_work = df_work.drop_duplicates(subset=[dt_col])
        df_work = df_work.set_index(dt_col)

        # Select target & numeric exogenous features
        numeric_cols = df_work.select_dtypes(include=[np.number]).columns.tolist()
        if target_col not in numeric_cols:
            df_work[target_col] = pd.to_numeric(df_work[target_col], errors="coerce")
            numeric_cols.append(target_col)

        df_work = df_work[numeric_cols]

        # Resample to ensure strictly regular time frequency
        raw_count = len(df_work)
        df_resampled = df_work.resample(resample_freq).mean()

        # Missing value treatment: linear interpolation followed by forward/backward fill
        interpolated_count = int(df_resampled[target_col].isna().sum())
        df_resampled = df_resampled.interpolate(method="time").ffill().bfill()

        # Outlier clipping
        outliers_detected = 0
        if clean_outliers:
            q25 = df_resampled[target_col].quantile(0.25)
            q75 = df_resampled[target_col].quantile(0.75)
            iqr = q75 - q25
            lower_bound = max(0.0, q25 - clip_iqr_multiplier * iqr)
            upper_bound = q75 + clip_iqr_multiplier * iqr

            outlier_mask = (df_resampled[target_col] < lower_bound) | (df_resampled[target_col] > upper_bound)
            outliers_detected = int(outlier_mask.sum())
            df_resampled[target_col] = df_resampled[target_col].clip(lower=lower_bound, upper=upper_bound)

        metadata = {
            "datetime_col": dt_col,
            "target_col": target_col,
            "region_col": region_col,
            "available_regions": available_regions,
            "selected_region": selected_region,
            "resample_freq": resample_freq,
            "raw_records": raw_count,
            "resampled_records": len(df_resampled),
            "interpolated_points": interpolated_count,
            "outliers_cleaned": outliers_detected,
            "start_time": df_resampled.index.min().isoformat(),
            "end_time": df_resampled.index.max().isoformat(),
            "mean_consumption": float(df_resampled[target_col].mean()),
            "max_consumption": float(df_resampled[target_col].max()),
            "min_consumption": float(df_resampled[target_col].min()),
            "std_consumption": float(df_resampled[target_col].std()),
        }

        return df_resampled, metadata
