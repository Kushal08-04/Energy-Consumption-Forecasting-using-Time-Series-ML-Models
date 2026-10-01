"""
Trainer, Cross-Validation, and Continuous Retraining Engine for Energy Forecasting.
Orchestrates training, walk-forward validation, recursive horizon forecasting,
model checkpointing, and continuous model updating.
"""

from typing import Dict, Any, List, Tuple, Optional
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.feature_engineering import TimeSeriesFeatureEngineer
from src.models import (
    BaseForecaster,
    XGBoostForecaster,
    RandomForestForecaster,
    OnlineIncrementalForecaster,
    HoltWintersForecaster,
    AdaptiveEnsembleForecaster,
)
from src.drift_detector import EnergyDriftDetector


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes standard time-series evaluation metrics."""
    eps = 1e-5
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(eps, y_true))) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE": round(mape, 2),
        "R2": round(r2, 4),
    }


class EnergyForecastingPipeline:
    """
    Complete end-to-end pipeline handling training, inference,
    continuous learning updates, and model registry management.
    """

    def __init__(
        self,
        target_col: str = "energy_consumption_mw",
        freq: str = "1h",
        registry_dir: str = "models/registry",
    ):
        self.target_col = target_col
        self.freq = freq
        self.registry_dir = registry_dir
        os.makedirs(self.registry_dir, exist_ok=True)

        self.feature_engineer = TimeSeriesFeatureEngineer(target_col=target_col, freq=freq)
        self.drift_detector = EnergyDriftDetector()

        self.models: Dict[str, BaseForecaster] = {}
        self.scorecard: pd.DataFrame = pd.DataFrame()
        self.best_model_name: str = "Adaptive Ensemble"
        self.last_train_data: Optional[pd.DataFrame] = None
        self.version_history: List[Dict[str, Any]] = self._load_version_history()

    def _get_history_file(self) -> str:
        return os.path.join(self.registry_dir, "version_history.json")

    def _load_version_history(self) -> List[Dict[str, Any]]:
        history_file = self._get_history_file()
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save_version_history(self):
        history_file = self._get_history_file()
        with open(history_file, "w") as f:
            json.dump(self.version_history, f, indent=2)

    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        test_horizon: int = 168,  # 7 days test window
        region_name: str = "Default-Region",
        trigger_type: str = "Initial Train",
    ) -> Dict[str, Any]:
        """
        Trains full suite of models using time-series split, evaluates them,
        calibrates the ensemble, and registers a version checkpoint.
        """
        self.last_train_data = df.copy()

        # Generate feature matrix
        df_feat = self.feature_engineer.create_features(df, is_training=True)
        X, y = self.feature_engineer.get_feature_matrix(df_feat)

        n = len(X)
        if n <= test_horizon + 24:
            test_horizon = max(24, int(n * 0.15))

        split_idx = n - test_horizon
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

        # Instantiate models
        xgb_model = XGBoostForecaster()
        rf_model = RandomForestForecaster()
        online_model = OnlineIncrementalForecaster()
        hw_model = HoltWintersForecaster()

        # Calibration split for ensemble
        calib_split = int(len(X_train) * 0.85)
        X_tr, y_tr = X_train.iloc[:calib_split], y_train.iloc[:calib_split]
        X_cal, y_cal = X_train.iloc[calib_split:], y_train.iloc[calib_split:]

        ensemble_model = AdaptiveEnsembleForecaster([
            XGBoostForecaster(),
            RandomForestForecaster(),
            OnlineIncrementalForecaster(),
        ])

        all_models = {
            "XGBoost": xgb_model,
            "Random Forest": rf_model,
            "Online Incremental (SGD)": online_model,
            "Holt-Winters (ETS)": hw_model,
            "Adaptive Ensemble": ensemble_model,
        }

        # Train & Evaluate each
        results = []
        fitted_models = {}
        test_predictions = {}

        for name, model in all_models.items():
            if name == "Adaptive Ensemble":
                model.fit_and_calibrate(X_tr, y_tr, X_cal, y_cal)
            else:
                model.fit(X_train, y_train)

            preds = model.predict(X_test)
            test_predictions[name] = preds
            metrics = compute_metrics(y_test.values, preds)
            metrics["Model"] = name
            results.append(metrics)
            fitted_models[name] = model

        scorecard = pd.DataFrame(results).set_index("Model")
        # Sort by MAPE ascending
        scorecard = scorecard.sort_values(by="MAPE", ascending=True)

        self.models = fitted_models
        self.scorecard = scorecard
        self.best_model_name = scorecard.index[0]

        # Drift assessment
        best_preds = test_predictions[self.best_model_name]
        drift_report = self.drift_detector.evaluate_drift(
            series=df[self.target_col],
            actuals=y_test.values,
            predictions=best_preds,
        )

        # Register version checkpoint
        new_version_num = len(self.version_history) + 1
        version_entry = {
            "version": f"v{new_version_num}.0",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "region": region_name,
            "trigger_type": trigger_type,
            "best_model": self.best_model_name,
            "best_mape": scorecard.loc[self.best_model_name, "MAPE"],
            "best_rmse": scorecard.loc[self.best_model_name, "RMSE"],
            "records_trained": len(df),
            "drift_status": drift_report["status"],
        }
        self.version_history.append(version_entry)
        self._save_version_history()

        return {
            "scorecard": scorecard,
            "best_model": self.best_model_name,
            "test_predictions": test_predictions,
            "test_actuals": y_test,
            "test_index": y_test.index,
            "drift_report": drift_report,
            "version_entry": version_entry,
        }

    def continuous_online_update(
        self,
        new_data_batch: pd.DataFrame,
        region_name: str = "Default-Region"
    ) -> Dict[str, Any]:
        """
        Incrementally updates online continuous models with streaming/new data batch.
        Uses partial_fit() to adapt weights without re-training from scratch.
        """
        if self.last_train_data is not None:
            combined = pd.concat([self.last_train_data, new_data_batch])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            self.last_train_data = combined
        else:
            self.last_train_data = new_data_batch.copy()

        df_feat = self.feature_engineer.create_features(self.last_train_data, is_training=True)
        X, y = self.feature_engineer.get_feature_matrix(df_feat)

        # Take the newest batch features
        batch_size = min(len(new_data_batch), len(X))
        X_batch, y_batch = X.iloc[-batch_size:], y.iloc[-batch_size:]

        # Update online models
        updated_models = []
        for name, model in self.models.items():
            if model.supports_online or isinstance(model, (OnlineIncrementalForecaster, AdaptiveEnsembleForecaster)):
                model.partial_fit(X_batch, y_batch)
                updated_models.append(name)

        # Re-evaluate drift
        drift_report = self.drift_detector.evaluate_drift(self.last_train_data[self.target_col])

        # Register continuous checkpoint
        new_version_num = len(self.version_history) + 1
        version_entry = {
            "version": f"v{new_version_num}.0-online",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "region": region_name,
            "trigger_type": "Online Streaming Update",
            "best_model": "Online Incremental / Ensemble",
            "best_mape": self.scorecard.loc[self.best_model_name, "MAPE"] if not self.scorecard.empty else 0.0,
            "best_rmse": self.scorecard.loc[self.best_model_name, "RMSE"] if not self.scorecard.empty else 0.0,
            "records_trained": len(self.last_train_data),
            "drift_status": drift_report["status"],
        }
        self.version_history.append(version_entry)
        self._save_version_history()

        return {
            "updated_models": updated_models,
            "batch_size": batch_size,
            "drift_report": drift_report,
            "version_entry": version_entry,
        }

    def forecast_future(
        self,
        df: pd.DataFrame,
        horizon_steps: int = 48,
        model_name: Optional[str] = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Recursively forecasts energy consumption for future steps.
        Produces point forecasts, upper bound, and lower bound.
        """
        if model_name is None or model_name not in self.models:
            model_name = self.best_model_name

        model = self.models[model_name]
        last_dt = df.index[-1]
        freq_offset = pd.tseries.frequencies.to_offset(self.freq)

        # Build future date range
        future_dates = pd.date_range(
            start=last_dt + freq_offset,
            periods=horizon_steps,
            freq=self.freq,
        )

        # Estimate residual uncertainty std
        if not self.scorecard.empty and model_name in self.scorecard.index:
            residual_std = float(self.scorecard.loc[model_name, "RMSE"])
        else:
            residual_std = float(df[self.target_col].std() * 0.06)

        # Working buffer for autoregressive recursive prediction
        buffer_df = df[[self.target_col]].copy()
        # Include temperature if available by carrying forward or periodic average
        if "temperature_c" in df.columns:
            temp_history = df["temperature_c"]
            buffer_df["temperature_c"] = temp_history

        forecast_records = []

        for step_idx, step_dt in enumerate(future_dates):
            # 1. Engineer features on the buffer up to this point
            df_feat = self.feature_engineer.create_features(buffer_df, is_training=False)
            latest_row = df_feat.iloc[[-1]]
            X_step = latest_row[self.feature_engineer.feature_names]

            # 2. Predict point forecast
            pred_val = float(model.predict(X_step)[0])
            pred_val = max(10.0, pred_val)

            # 3. Uncertainty growth over horizon
            # Standard error grows with sqrt(1 + step / 12)
            horizon_scale = np.sqrt(1.0 + (step_idx / 24.0))
            z_score = 1.96 if confidence_level >= 0.95 else 1.645
            margin = z_score * residual_std * horizon_scale

            lower_bound = max(0.0, pred_val - margin)
            upper_bound = pred_val + margin

            forecast_records.append({
                "timestamp": step_dt,
                "forecast_mw": round(pred_val, 2),
                "lower_bound_mw": round(lower_bound, 2),
                "upper_bound_mw": round(upper_bound, 2),
                "uncertainty_margin": round(margin, 2),
            })

            # 4. Append predicted point into buffer for subsequent lag computation
            new_row = {self.target_col: pred_val}
            if "temperature_c" in buffer_df.columns:
                # Use daily diurnal temperature approximation
                h = step_dt.hour
                base_temp = float(buffer_df["temperature_c"].iloc[-24]) if len(buffer_df) >= 24 else 20.0
                new_row["temperature_c"] = base_temp + 3.0 * np.sin(2 * np.pi * (h - 9) / 24)

            new_df_row = pd.DataFrame(new_row, index=[step_dt])
            buffer_df = pd.concat([buffer_df, new_df_row])

        return pd.DataFrame(forecast_records).set_index("timestamp")
