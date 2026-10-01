"""
Script to test CSV and PDF report generation directly to disk.
"""

import os
import pandas as pd
from src.data_loader import EnergyDataLoader
from src.trainer import EnergyForecastingPipeline
from src.pdf_generator import EnergyForecastPDFReport

def main():
    print("Testing data load & training...")
    df = pd.read_csv(os.path.join("data", "sample_energy_data.csv"))
    loader = EnergyDataLoader()
    processed_df, meta = loader.process(df, selected_region="North-Metro")

    pipeline = EnergyForecastingPipeline()
    res = pipeline.train_and_evaluate(processed_df, test_horizon=48, region_name="North-Metro")
    print(f"Trained models. Best model: {res['best_model']}")
    print(res["scorecard"])

    print("Generating 48h forecast...")
    forecast_df = pipeline.forecast_future(processed_df, horizon_steps=48)
    
    os.makedirs("exports", exist_ok=True)
    csv_path = os.path.join("exports", "test_forecast.csv")
    pdf_path = os.path.join("exports", "test_report.pdf")

    forecast_df.to_csv(csv_path)
    print(f"Saved forecast CSV: {csv_path} ({os.path.getsize(csv_path)} bytes)")

    print("Generating PDF report...")
    pdf_gen = EnergyForecastPDFReport()
    pdf_bytes = pdf_gen.generate_pdf(
        historical_df=processed_df,
        forecast_df=forecast_df,
        scorecard_df=res["scorecard"],
        metadata={**meta, "selected_region": "North-Metro", "best_model": res["best_model"]},
        drift_report=res["drift_report"],
        output_path=pdf_path
    )
    print(f"Saved forecast PDF: {pdf_path} ({os.path.getsize(pdf_path)} bytes)")

if __name__ == "__main__":
    main()
