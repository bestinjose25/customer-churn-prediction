import unittest

import pandas as pd
from sklearn.linear_model import LogisticRegression

from predict import score_customers
from train import make_pipeline, prepare_features


class BatchPredictionTests(unittest.TestCase):
    def setUp(self):
        training = pd.DataFrame({
            "customerID": ["a", "b", "c", "d"],
            "TotalCharges": [" ", "20", "10", "40"],
            "Contract": ["A", "B", "A", "B"],
        })
        features = prepare_features(training)
        self.pipeline = make_pipeline(features, LogisticRegression()).fit(features, [0, 1, 0, 1])
        self.incoming = pd.DataFrame({
            "Contract": ["unseen", "A"],
            "customerID": ["002", "001"],
            "TotalCharges": [" ", "10"],
        }, index=[2, 8])

    def test_scores_unlabeled_rows_in_input_order(self):
        result = score_customers(self.incoming, self.pipeline)
        self.assertEqual(result["customerID"].tolist(), ["002", "001"])
        self.assertTrue(result["churn_probability"].between(0, 1).all())
        self.assertEqual(
            result["predicted_churn"].tolist(),
            ["Yes" if probability >= 0.5 else "No" for probability in result["churn_probability"]],
        )

    def test_reports_missing_model_feature(self):
        with self.assertRaisesRegex(ValueError, "Missing model features.*Contract"):
            score_customers(self.incoming.drop(columns="Contract"), self.pipeline)
