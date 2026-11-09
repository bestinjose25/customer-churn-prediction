"""Train and save a churn pipeline. Run python train.py --help."""
import argparse
import json
from pathlib import Path
import platform

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path):
    data = pd.read_csv(path)
    required = {"Churn", "customerID", "TotalCharges"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(data.columns))}")
    if data["customerID"].isna().any() or data["customerID"].duplicated().any():
        raise ValueError("Customer IDs must be present and unique.")
    target = data["Churn"].map({"No": 0, "Yes": 1})
    if target.isna().any() or target.nunique() != 2:
        raise ValueError("Churn must contain both Yes and No, without missing labels.")
    features = data.drop(columns=["Churn", "customerID"]).copy()
    # Coerce whitespace/missing charges before numerical conversion; impute in training.
    features["TotalCharges"] = pd.to_numeric(features["TotalCharges"], errors="coerce")
    return features, target


def make_pipeline(features, model):
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocessing", preprocessing), ("model", model)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    features, target = load_data(args.data)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=args.seed)
    models = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=args.seed),
        "balanced_logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed),
        "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=args.seed, n_jobs=1),
    }
    results, pipelines = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for name, model in models.items():
        pipeline = make_pipeline(x_train, model)
        scores = cross_validate(pipeline, x_train, y_train, cv=cv,
                                scoring={"ap": "average_precision", "recall": "recall"})
        results[name] = {"cv_average_precision": float(scores["test_ap"].mean()),
                         "cv_average_precision_std": float(scores["test_ap"].std()),
                         "cv_recall": float(scores["test_recall"].mean())}
        pipelines[name] = pipeline
    # Select using training CV only; evaluate the chosen model once on held-out data.
    winner = max(results, key=lambda name: results[name]["cv_average_precision"])
    pipeline = pipelines[winner].fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    test = {"accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions),
            "f1": f1_score(y_test, predictions),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "average_precision": average_precision_score(y_test, probabilities),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist()}
    report = {"seed": args.seed, "train_rows": len(x_train), "test_rows": len(x_test),
              "selected_model": winner, "selection_metric": "cv_average_precision",
              "threshold": 0.5, "cross_validation": results, "test": test,
              "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__, "pandas": pd.__version__}}
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, args.output / "churn_pipeline.joblib")
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
