#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
載入 model.pkl + 測試資料，輸出報告 & 混淆矩陣
用法：
python src/evaluate.py --model output/model.pkl --test data/processed/features.csv --report output/report.txt
"""
import argparse, joblib, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate(model_path: str, test_csv: str, report_path: str) -> None:
    model = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    X_test, y_test = df.drop('label', axis=1), df['label']
    y_pred = model.predict(X_test)

    # 文字報告
    report_txt  = classification_report(y_test, y_pred)
    cm          = confusion_matrix(y_test, y_pred)
    with open(report_path, "w") as f:
        f.write("=== Classification Report ===\n")
        f.write(report_txt)
        f.write("\n=== Confusion Matrix ===\n")
        f.write(cm.__repr__())
    print(" Saved report to", report_path)

    # 圖形混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['normal','abnormal'])
    disp.plot(cmap=plt.cm.YlOrRd, values_format='g')
    plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()
    evaluate(args.model, args.test, args.report)
