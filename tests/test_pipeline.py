"""
Unit & Integration Tests for Energy Consumption Forecasting Pipeline.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.data_loader import generate_benchmark_energy_data, EnergyDataLoader
from src.feature_engineering import TimeSeriesFeatureEngineer
from src.models import (
    XGBoostForecaster,
    RandomForestForecaster,
    OnlineIncrementalForecaster,
    AdaptiveEnsembleForecaster,
)
from src.drift_detector import EnergyDriftDetector
from src.trainer import EnergyForecastingPipeline
from src.pdf_generator import EnergyForecastPDFReport


class TestEnergyForecastingPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate small dataset for testing
        cls.raw_df = generate_benchmark_energy_data(
            start_date="2025-01-01",
            end_date="2025-02-15",
            freq="1h",
            random_seed=42
        )
        loader = EnergyDataLoader()
        cls.processed_df, cls.metadata = loader.process(cls.raw_df, selected_region="North-Metro")

    def test_01_data_loader(self):
        self.assertFalse(self.processed_df.empty)
        self.assertIn("energy_consumption_mw", self.processed_df.columns)
        self.assertTrue(isinstance(self.processed_df.index, pd.DatetimeIndex))
        self.assertGreater(self.metadata["resampled_records"], 500)

    def test_02_feature_engineering(self):
        fe = TimeSeriesFeatureEngineer(target_col="energy_consumption_mw", freq="1h")
        feat_df = fe.create_features(self.processed_df, is_training=True)
        self.assertIn("hour_sin", feat_df.columns)
        self.assertIn("lag_1", feat_df.columns)
        self.assertIn("lag_24", feat_df.columns)
        self.assertIn("rolling_mean_24", feat_df.columns)
        self.assertFalse(feat_df.isna().any().any())

    def test_03_models_and_online_learning(self):
        fe = TimeSeriesFeatureEngineer(target_col="energy_consumption_mw", freq="1h")
        feat_df = fe.create_features(self.processed_df, is_training=True)
        X, y = fe.get_feature_matrix(feat_df)

        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]
        X_test, y_test = X.iloc[split:], y.iloc[split:]

        # Test XGBoost
        xgb = XGBoostForecaster(n_estimators=20)
        xgb.fit(X_train, y_train)
        p_xgb = xgb.predict(X_test)
        self.assertEqual(len(p_xgb), len(y_test))

        # Test Online Incremental Learner & partial_fit
        online = OnlineIncrementalForecaster()
        online.fit(X_train.iloc[:100], y_train.iloc[:100])
        initial_preds = online.predict(X_test)
        # Update with new stream chunk
        online.partial_fit(X_train.iloc[100:200], y_train.iloc[100:200])
        updated_preds = online.predict(X_test)
        self.assertEqual(len(updated_preds), len(y_test))
        # Ensure weights adjusted
        self.assertEqual(online.update_count, 2)

    def test_04_drift_detection(self):
        detector = EnergyDriftDetector(reference_window_size=200, recent_window_size=72)
        series = self.processed_df["energy_consumption_mw"]
        res = detector.evaluate_drift(series)
        self.assertIn("status", res)
        self.assertIn("ks_statistic", res)
        self.assertIn("p_value", res)

    def test_05_end_to_end_training_and_forecasting(self):
        pipeline = EnergyForecastingPipeline(target_col="energy_consumption_mw", freq="1h")
        train_res = pipeline.train_and_evaluate(
            df=self.processed_df,
            test_horizon=48,
            region_name="North-Metro",
            trigger_type="Test Run"
        )
        self.assertIn("scorecard", train_res)
        self.assertGreater(len(train_res["scorecard"]), 0)

        # Forecast future 24h
        future_df = pipeline.forecast_future(self.processed_df, horizon_steps=24)
        self.assertEqual(len(future_df), 24)
        self.assertIn("forecast_mw", future_df.columns)
        self.assertIn("lower_bound_mw", future_df.columns)
        self.assertIn("upper_bound_mw", future_df.columns)
        # Upper bound should be >= lower bound
        self.assertTrue((future_df["upper_bound_mw"] >= future_df["lower_bound_mw"]).all())

        # Test continuous online batch update
        new_batch = self.processed_df.iloc[-48:].copy()
        online_res = pipeline.continuous_online_update(new_batch, region_name="North-Metro")
        self.assertIn("updated_models", online_res)

        # Test PDF Generation
        pdf_gen = EnergyForecastPDFReport()
        pdf_bytes = pdf_gen.generate_pdf(
            historical_df=self.processed_df,
            forecast_df=future_df,
            scorecard_df=train_res["scorecard"],
            metadata=self.metadata,
            drift_report=train_res["drift_report"],
        )
        self.assertGreater(len(pdf_bytes), 1000)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
