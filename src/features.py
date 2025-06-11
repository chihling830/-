#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特徵工程：
* One-Hot 編碼類別型
* Min-Max Scaler 數值型
用法：
python src/features.py --input data/processed/cleaned.csv --output data/processed/features.csv
"""
import argparse, pandas as pd
from sklearn.preprocessing import MinMaxScaler

def engineer(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    # 1️⃣ One-Hot
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # 2️⃣ Min-Max
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    df.to_csv(output_path, index=False)
    print(f" Saved feature data to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    engineer(args.input, args.output)
