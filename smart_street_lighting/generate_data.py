"""
Utility script to generate the realistic sample dataset with 650+ records.
Can be run standalone: python generate_data.py
"""

import os
import pandas as pd
from modules.preprocessing import generate_sample_dataset

def main():
    print("Generating realistic Smart Street Lighting dataset (650+ records)...")
    df = generate_sample_dataset(n_records=650)
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "sample_street_lights.csv")
    df.to_csv(out_path, index=False)
    print(f"Successfully wrote {len(df)} records to {out_path}")

if __name__ == "__main__":
    main()
