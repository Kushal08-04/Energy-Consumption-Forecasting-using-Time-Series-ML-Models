"""
Time-Series Forecasting Models Suite.
Includes XGBoost, Random Forest, Online Incremental Learner (SGD), Holt-Winters,
and an Adaptive Performance-Weighted Ensemble.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb


class BaseForecaster(ABC):
    """Abstract base class for all forecasters."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self.supports_online = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseForecaster":
        """Incremental online weight update. Default raises NotImplementedError."""
        raise NotImplementedError(f"{self.name} does not support online incremental updates.")


class XGBoostForecaster(BaseForecaster):
    """High-accuracy Gradient Boosted Decision Tree Forecaster."""

    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.04,
        max_depth: int = 6,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
    ):
        super().__init__(name="XGBoost")
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostForecaster":
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted yet.")
        preds = self.model.predict(X[self.feature_names_])
        return np.maximum(0.0, preds)

    def get_feature_importances(self) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(dtype=float)
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)


class RandomForestForecaster(BaseForecaster):
    """Robust Bagged Random Forest Forecaster."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 12,
        min_samples_leaf: int = 3,
        random_state: int = 42,
    ):
        super().__init__(name="Random Forest")
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestForecaster":
        self.feature_names_ = list(X.columns)
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted yet.")
        preds = self.model.predict(X[self.feature_names_])
        return np.maximum(0.0, preds)

    def get_feature_importances(self) -> pd.Series:
        if not self.is_fitted:
            return pd.Series(dtype=float)
        return pd.Series(self.model.feature_importances_, index=self.feature_names_).sort_values(ascending=False)


class OnlineIncrementalForecaster(BaseForecaster):
    """
    True Online Continuous Learning Forecaster using SGDRegressor.
    Supports incremental weight updates via partial_fit() on streaming data chunks.
    Prevents data decay with continuous adaptation.
    """

    def __init__(
        self,
        alpha: float = 1e-4,
        eta0: float = 0.01,
        learning_rate: str = "adaptive",
        random_state: int = 42,
    ):
        super().__init__(name="Online Incremental (SGD)")
        self.supports_online = True
        self.scaler = StandardScaler()
        self.model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=alpha,
            eta0=eta0,
            learning_rate=learning_rate,
            random_state=random_state,
            max_iter=2000,
            tol=1e-3,
            warm_start=True,
        )
        self.feature_names_: List[str] = []
        self.update_count: int = 0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OnlineIncrementalForecaster":
        self.feature_names_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.update_count = 1
        return self

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> "OnlineIncrementalForecaster":
        """Incrementally updates weights on a newly arrived batch of streaming records."""
        if not self.is_fitted:
            return self.fit(X, y)

        X_aligned = X[self.feature_names_]
        # Incremental scaling
        self.scaler.partial_fit(X_aligned)
        X_scaled = self.scaler.transform(X_aligned)
        self.model.partial_fit(X_scaled, y)
        self.update_count += 1
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted yet.")
        X_scaled = self.scaler.transform(X[self.feature_names_])
        preds = self.model.predict(X_scaled)
        return np.maximum(0.0, preds)


class HoltWintersForecaster(BaseForecaster):
    """Statistical Baseline Forecaster using Triple Exponential Smoothing."""

    def __init__(self, seasonal_periods: int = 24, trend: str = "add", seasonal: str = "add"):
        super().__init__(name="Holt-Winters (ETS)")
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal
        self.fitted_model_ = None
        self.last_timestamp_ = None
        self.freq_ = "1h"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HoltWintersForecaster":
        self.last_timestamp_ = y.index[-1]
        try:
            model = ExponentialSmoothing(
                y.values,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method="estimated",
            )
            self.fitted_model_ = model.fit(optimized=True)
            self.is_fitted = True
        except Exception:
            # Fallback to simple additive without seasonal if series is short
            model = ExponentialSmoothing(y.values, trend="add", seasonal=None)
            self.fitted_model_ = model.fit(optimized=True)
            self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.fitted_model_ is None:
            raise ValueError(f"{self.name} is not fitted yet.")
        steps = len(X)
        preds = self.fitted_model_.forecast(steps)
        return np.maximum(0.0, np.array(preds))


class AdaptiveEnsembleForecaster(BaseForecaster):
    """
    Weighted Ensemble combining predictions of multiple candidate models.
    Weights are inversely proportional to out-of-sample validation MAPE/RMSE.
    """

    def __init__(self, models: Optional[List[BaseForecaster]] = None):
        super().__init__(name="Adaptive Ensemble")
        if models is None:
            self.models = [
                XGBoostForecaster(),
                RandomForestForecaster(),
                OnlineIncrementalForecaster(),
            ]
        else:
            self.models = models
        self.weights_: Dict[str, float] = {}

    def fit_and_calibrate(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> "AdaptiveEnsembleForecaster":
        """Fits all models on train set and computes optimal weights on validation set."""
        errors = {}
        for m in self.models:
            m.fit(X_train, y_train)
            preds_val = m.predict(X_val)
            # Use MAPE for weighting
            mape = np.mean(np.abs((y_val.values - preds_val) / np.maximum(1e-5, y_val.values)))
            errors[m.name] = max(1e-4, mape)

        # Inverse error weighting
        inv_errors = {name: 1.0 / err for name, err in errors.items()}
        total_inv = sum(inv_errors.values())
        self.weights_ = {name: inv / total_inv for name, inv in inv_errors.items()}
        self.is_fitted = True
        return self

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaptiveEnsembleForecaster":
        """Standard fit: splits last 15% as calibration validation set."""
        n = len(X)
        split_idx = int(n * 0.85)
        X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
        X_val, y_val = X.iloc[split_idx:], y.iloc[split_idx:]
        return self.fit_and_calibrate(X_train, y_train, X_val, y_val)

    def partial_fit(self, X: pd.DataFrame, y: pd.Series) -> "AdaptiveEnsembleForecaster":
        """Propagates incremental update to any models supporting online learning."""
        for m in self.models:
            if m.supports_online:
                m.partial_fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError(f"{self.name} is not fitted yet.")
        ensemble_pred = np.zeros(len(X))
        for m in self.models:
            w = self.weights_.get(m.name, 1.0 / len(self.models))
            preds = m.predict(X)
            ensemble_pred += w * preds
        return np.maximum(0.0, ensemble_pred)
