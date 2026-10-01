"""
Utility script to generate sample multi-region energy dataset.
"""
import os
from src.data_loader import generate_benchmark_energy_data

def main():
    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "sample_energy_data.csv")
    print("Generating benchmark energy dataset...")
    df = generate_benchmark_energy_data(
        start_date="2023-01-01",
        end_date="2026-03-31",
        freq="1h",
        random_seed=42
    )
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} records across {df['region'].nunique()} regions to {out_path}")

if __name__ == "__main__":
    main()
