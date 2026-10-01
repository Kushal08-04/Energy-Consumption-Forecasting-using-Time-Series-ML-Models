"""
Test to verify custom dataset upload with arbitrary column names.
"""

import unittest
import numpy as np
import pandas as pd
from src.data_loader import EnergyDataLoader
from src.trainer import EnergyForecastingPipeline
from src.pdf_generator import EnergyForecastPDFReport


class TestCustomUpload(unittest.TestCase):

    def test_arbitrary_column_names(self):
        # Create a synthetic custom dataset with completely arbitrary column names
        dates = pd.date_range("2024-01-01", periods=500, freq="1h")
        y = 5000 + 800 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 50, len(dates))

        custom_df = pd.DataFrame({
            "Time_Recorded": dates,
            "MegaWatts_Demand": y,
            "City_Zone": ["East"] * len(dates),
        })

        # 1. Loader with custom column mapping
        loader = EnergyDataLoader(target_col="MegaWatts_Demand", datetime_col="Time_Recorded")
        processed, meta = loader.process(custom_df)

        self.assertEqual(meta["target_col"], "MegaWatts_Demand")
        self.assertIn("MegaWatts_Demand", processed.columns)

        # 2. Pipeline training
        pipeline = EnergyForecastingPipeline(target_col="MegaWatts_Demand", freq="1h")
        train_res = pipeline.train_and_evaluate(
            df=processed,
            test_horizon=24,
            region_name="East",
            target_col="MegaWatts_Demand",
        )
        self.assertIn("scorecard", train_res)

        # 3. Forecast future
        forecast_df = pipeline.forecast_future(
            df=processed,
            horizon_steps=24,
            target_col="MegaWatts_Demand",
        )
        self.assertEqual(len(forecast_df), 24)
        self.assertIn("forecast_mw", forecast_df.columns)

        # 4. Online update
        new_batch = processed.iloc[-24:].copy()
        online_res = pipeline.continuous_online_update(
            new_batch,
            region_name="East",
            target_col="MegaWatts_Demand",
        )
        self.assertIn("updated_models", online_res)

        # 5. PDF generation
        pdf_gen = EnergyForecastPDFReport()
        pdf_bytes = pdf_gen.generate_pdf(
            historical_df=processed,
            forecast_df=forecast_df,
            scorecard_df=train_res["scorecard"],
            metadata=meta,
            drift_report=train_res["drift_report"],
        )
        self.assertTrue(len(pdf_bytes) > 1000)


if __name__ == "__main__":
    unittest.main()
