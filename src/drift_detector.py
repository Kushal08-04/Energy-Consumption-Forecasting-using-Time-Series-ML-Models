"""
Concept Drift & Covariate Shift Detector for Energy Consumption Time Series.
Monitors distribution shifts and error degradation to trigger continuous retraining.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class EnergyDriftDetector:
    """
    Monitors energy consumption distributions and model prediction errors
    to detect statistical concept drift and data decay.
    """

    def __init__(
        self,
        reference_window_size: int = 720,  # 30 days of hourly data
        recent_window_size: int = 168,     # 7 days of hourly data
        p_val_threshold: float = 0.01,
        mape_degradation_ratio: float = 1.35,
    ):
        self.reference_window_size = reference_window_size
        self.recent_window_size = recent_window_size
        self.p_val_threshold = p_val_threshold
        self.mape_degradation_ratio = mape_degradation_ratio

    def evaluate_drift(
        self,
        series: pd.Series,
        actuals: Optional[np.ndarray] = None,
        predictions: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Assesses both distribution shift (KS-test) and performance degradation.
        """
        n = len(series)
        if n < self.recent_window_size * 2:
            return {
                "status": "Healthy",
                "message": "Insufficient data to establish reference window.",
                "ks_statistic": 0.0,
                "p_value": 1.0,
                "drift_detected": False,
                "mape_ratio": 1.0,
            }

        # Reference distribution vs Recent distribution
        ref_size = min(self.reference_window_size, n - self.recent_window_size)
        ref_data = series.iloc[:ref_size].values
        recent_data = series.iloc[-self.recent_window_size:].values

        # Two-sample Kolmogorov-Smirnov test
        ks_res = ks_2samp(ref_data, recent_data)
        ks_stat = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # Performance error degradation check
        mape_ratio = 1.0
        perf_drift = False
        if actuals is not None and predictions is not None and len(actuals) == len(predictions):
            eps = 1e-5
            errors = np.abs((actuals - predictions) / np.maximum(eps, actuals)) * 100.0
            if len(errors) >= self.recent_window_size * 2:
                ref_err = float(np.mean(errors[:ref_size]))
                recent_err = float(np.mean(errors[-self.recent_window_size:]))
                mape_ratio = round(recent_err / max(1e-3, ref_err), 2)
                perf_drift = mape_ratio > self.mape_degradation_ratio

        # Determine drift flag
        dist_drift = (p_val < self.p_val_threshold) and (ks_stat > 0.15)
        drift_detected = dist_drift or perf_drift

        if drift_detected:
            status = "Drift Detected (Retrain Required)"
            if dist_drift and perf_drift:
                msg = f"Severe drift: Distribution shifted (KS={ks_stat:.3f}, p={p_val:.4f}) and MAPE degraded by {mape_ratio}x."
            elif dist_drift:
                msg = f"Distributional shift detected in recent energy consumption (KS={ks_stat:.3f}, p={p_val:.4f})."
            else:
                msg = f"Model accuracy degradation detected: Recent error is {mape_ratio}x of baseline."
        elif ks_stat > 0.10:
            status = "Warning: Minor Shift"
            msg = f"Minor distribution variation noticed (KS={ks_stat:.3f}). Keep monitoring."
        else:
            status = "Healthy (Optimal)"
            msg = "Data distribution and forecast error metrics are stable. No data decay."

        return {
            "status": status,
            "message": msg,
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_val, 6),
            "drift_detected": drift_detected,
            "mape_ratio": mape_ratio,
            "ref_mean": round(float(np.mean(ref_data)), 2),
            "recent_mean": round(float(np.mean(recent_data)), 2),
            "ref_std": round(float(np.std(ref_data)), 2),
            "recent_std": round(float(np.std(recent_data)), 2),
        }
