#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
訓練模型並輸出 model.pkl
用法：
python src/train_model.py --train data/processed/features.csv --model output/model.pkl --algo rf
"""
import argparse, joblib, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# 模型們
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

ALGO_MAP = {
    "logit": LogisticRegression(max_iter=500, class_weight={0:0.001, 1:0.999}),
    "svm"  : SVC(kernel="rbf", class_weight={0:0.001, 1:0.999}, probability=True),
    "rf"   : RandomForestClassifier(n_estimators=300, max_depth=None, random_state=1234, class_weight={0:0.001,1:0.999}),
    "xgb"  : XGBClassifier(max_depth=4, n_estimators=300, learning_rate=0.1, scale_pos_weight=100) if XGBClassifier else None
}

def train(train_csv: str, model_path: str, algo: str) -> None:
    if algo not in ALGO_MAP or ALGO_MAP[algo] is None:
        raise ValueError(f"algo 必須是 {list(ALGO_MAP.keys())}")

    df = pd.read_csv(train_csv)
    X, y = df.drop('label', axis=1), df['label']

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.3,
                                                random_state=1234, stratify=y)

    model = ALGO_MAP[algo]

    # GridSearch 範例（logit C 值）
    if algo == "logit":
        param = {"C":[0.01,0.1,1,10,100]}
        model = GridSearchCV(model, param, cv=5, scoring="f1")

    model.fit(X_tr, y_tr)
    print(" validation report\n",
          classification_report(y_val, model.predict(X_val)))
    joblib.dump(model, model_path)
    print(f" Saved model to {model_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--algo", default="rf", choices=list(ALGO_MAP.keys()))
    args = ap.parse_args()
    train(args.train, args.model, args.algo)
