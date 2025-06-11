#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
前處理：
1. 刪除缺失率 > 80% 欄位
2. 特定欄位缺值補 0
3. 其他數值缺值補中位數
用法：
python src/data_preprocess.py --input data/raw/raw.csv --output data/processed/cleaned.csv
"""
import argparse, pandas as pd

def preprocess(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    # 刪掉高缺失欄
    thresh = int(len(df) * 0.2)
    df = df.dropna(thresh=thresh, axis=1)

    # 指定欄補 0
    for col in ['每月薪津扣款比例']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 其餘數值補中位數
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    preprocess(args.input, args.output)
