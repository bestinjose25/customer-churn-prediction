"""Score a customer CSV using a trusted model saved by train.py."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from train import prepare_features


def score_customers(data, pipeline):
    if data.empty:
        raise ValueError("The input CSV must contain at least one customer.")
    features = prepare_features(data)
    expected = list(pipeline.feature_names_in_)
    missing = set(expected) - set(features.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")

    features = features[expected]
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = pipeline.predict(features)
    return pd.DataFrame({
        "customerID": data["customerID"],
        "churn_probability": probabilities,
        "predicted_churn": pd.Series(predictions, index=data.index).map({0: "No", 1: "Yes"}),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn_pipeline.joblib"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    args = parser.parse_args()

    if args.output.resolve() in {args.data.resolve(), args.model.resolve()}:
        parser.error("--output must be a different file from --data and --model")
    try:
        data = pd.read_csv(args.data, dtype={"customerID": str})
        predictions = score_customers(data, joblib.load(args.model))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {len(predictions):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
