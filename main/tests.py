"""
Surge Heartly - Test Suite
===========================
Covers three layers:
  1. Data loading & sanity checks  (TestDataPipeline)
  2. Model training & inference    (TestModelTraining)
  3. Django view integration       (TestPredictionView)

Run with:
    .\\venv\\Scripts\\python.exe manage.py test main --verbosity=2
"""

import os
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse

from main.prediction_system import (
    load_heart_attack_data,
    train_logistic_regression,
    train_random_forest,
    train_svm,
)
from main.neural_network import train_neural_network, predict


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# A synthetic high-risk patient (67yo male, textbook risk factors)
HIGH_RISK_POST = dict(
    age=67, sex=1, cp=3, trestbps=160, chol=286,
    fbs=1, restecg=2, thalach=108, exang=1,
    oldpeak=1.5, slope=1, ca=3, thal=2,
)

# A synthetic low-risk patient (35yo female, all benign values)
LOW_RISK_POST = dict(
    age=35, sex=0, cp=2, trestbps=115, chol=190,
    fbs=0, restecg=0, thalach=185, exang=0,
    oldpeak=0.0, slope=0, ca=0, thal=0,
)


# ---------------------------------------------------------------------------
# 1. Data pipeline tests
# ---------------------------------------------------------------------------
class TestDataPipeline(TestCase):
    """Verify that heart.csv loads into the correct shape and types."""

    def setUp(self):
        self.X, self.y = load_heart_attack_data()

    def test_sample_count(self):
        """Dataset must have exactly 303 samples."""
        self.assertEqual(
            self.X.shape[0], 303,
            "Expected 303 rows in heart.csv"
        )

    def test_feature_count(self):
        """Each sample must have exactly 13 features."""
        self.assertEqual(
            self.X.shape[1], 13,
            "Expected 13 feature columns"
        )

    def test_binary_labels(self):
        """Target column must contain only 0 and 1."""
        unique = set(np.unique(self.y))
        self.assertSetEqual(unique, {0, 1},
                            "Labels must be strictly binary (0 or 1)")

    def test_no_nan_in_features(self):
        """Feature matrix must contain no NaN values."""
        self.assertFalse(
            np.isnan(self.X).any(),
            "NaN values found in feature matrix"
        )

    def test_no_nan_in_labels(self):
        """Label vector must contain no NaN values."""
        self.assertFalse(
            np.isnan(self.y.astype(float)).any(),
            "NaN values found in label vector"
        )

    def test_label_class_balance(self):
        """
        Cleveland dataset is roughly balanced (~54% positive).
        Guard against an accidentally inverted target column.
        """
        pos_ratio = np.sum(self.y == 1) / len(self.y)
        self.assertGreater(pos_ratio, 0.40,
                           "Positive class ratio unexpectedly low")
        self.assertLess(pos_ratio, 0.60,
                        "Positive class ratio unexpectedly high")


# ---------------------------------------------------------------------------
# 2. Model training & inference tests
# ---------------------------------------------------------------------------
class TestModelTraining(TestCase):
    """
    Verify that each model trains without error and produces
    well-formed predictions on held-out inputs.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.X, cls.y = load_heart_attack_data()
        # Train all three sklearn models on the full dataset
        cls.lr  = train_logistic_regression(cls.X, cls.y)
        cls.rf  = train_random_forest(cls.X, cls.y)
        cls.svm = train_svm(cls.X, cls.y)
        # Train the MLP (also saves scaler to disk)
        cls.mlp = train_neural_network(cls.X, cls.y)

    # --- sklearn models: binary output check ---

    def _assert_binary_prediction(self, model, name):
        """A prediction on a single row must be 0 or 1."""
        sample = self.X[:1]          # shape (1, 13)
        pred = int(model.predict(sample)[0])
        self.assertIn(pred, [0, 1],
                      f"{name}.predict() returned non-binary value: {pred}")

    def test_logistic_regression_predicts(self):
        self._assert_binary_prediction(self.lr, "LogisticRegression")

    def test_random_forest_predicts(self):
        self._assert_binary_prediction(self.rf, "RandomForestClassifier")

    def test_svm_predicts(self):
        self._assert_binary_prediction(self.svm, "SVC")

    # --- sklearn models: probability output check ---

    def _assert_probability_output(self, model, name):
        """predict_proba() must return values in [0, 1] for both classes."""
        sample = self.X[:1]
        proba = model.predict_proba(sample)[0]
        self.assertEqual(len(proba), 2,
                         f"{name} must output probabilities for 2 classes")
        self.assertAlmostEqual(sum(proba), 1.0, places=5,
                               msg=f"{name} class probabilities must sum to 1")
        for p in proba:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_logistic_regression_probability(self):
        self._assert_probability_output(self.lr, "LogisticRegression")

    def test_random_forest_probability(self):
        self._assert_probability_output(self.rf, "RandomForestClassifier")

    def test_svm_probability(self):
        self._assert_probability_output(self.svm, "SVC")

    # --- MLP wrapper ---

    def test_mlp_predict_output_types(self):
        """
        neural_network.predict() must return (int, float) where
        the int is in {0, 1} and the float is in [0, 100].
        """
        v = HIGH_RISK_POST
        result, prob = predict(
            self.mlp,
            v["age"], v["sex"], v["cp"], v["trestbps"], v["chol"],
            v["fbs"], v["restecg"], v["thalach"], v["exang"],
            v["oldpeak"], v["slope"], v["ca"], v["thal"],
        )
        self.assertIn(result, [0, 1],
                      f"predict() result must be 0 or 1, got {result}")
        self.assertGreaterEqual(prob, 0.0,
                                f"Probability must be >= 0, got {prob}")
        self.assertLessEqual(prob, 100.0,
                             f"Probability must be <= 100, got {prob}")

    def test_mlp_predict_deterministic(self):
        """Calling predict() twice with identical input must return same output."""
        v = LOW_RISK_POST
        args = (
            self.mlp,
            v["age"], v["sex"], v["cp"], v["trestbps"], v["chol"],
            v["fbs"], v["restecg"], v["thalach"], v["exang"],
            v["oldpeak"], v["slope"], v["ca"], v["thal"],
        )
        r1, p1 = predict(*args)
        r2, p2 = predict(*args)
        self.assertEqual(r1, r2, "predict() result is non-deterministic")
        self.assertAlmostEqual(p1, p2, places=4,
                               msg="predict() probability is non-deterministic")


# ---------------------------------------------------------------------------
# 3. Django view integration tests
# ---------------------------------------------------------------------------
class TestPredictionView(TestCase):
    """
    Integration tests for the heart_attack_prediction view.

    Uses RequestFactory (not the full test Client) to bypass a known
    Django 4.2 / Python 3.14 incompatibility where Context.__copy__
    raises AttributeError during template-instrumented responses.

    RequestFactory calls the view function directly and returns the raw
    HttpResponse, letting us inspect status codes and rendered content
    without the broken copy machinery.
    """

    def setUp(self):
        from django.test import RequestFactory
        from main.views import heart_attack_prediction
        self.factory = RequestFactory()
        self.view = heart_attack_prediction

    def _get(self):
        request = self.factory.get("/")
        return self.view(request)

    def _post(self, data):
        # POST values must be strings, mirroring how browsers submit forms
        str_data = {k: str(v) for k, v in data.items()}
        request = self.factory.post("/", str_data)
        return self.view(request)

    # --- GET ---

    def test_get_returns_200(self):
        """GET / must return HTTP 200."""
        response = self._get()
        self.assertEqual(response.status_code, 200)

    def test_get_renders_form_tag(self):
        """GET / rendered HTML must contain a <form> element."""
        response = self._get()
        self.assertIn(b"<form", response.content)

    def test_get_contains_navbar_branding(self):
        """GET / must include 'Surge Heartly' in the rendered output."""
        response = self._get()
        self.assertIn(b"Surge Heartly", response.content)

    # --- POST with valid data ---

    def test_post_high_risk_returns_200(self):
        """Valid POST (high-risk profile) must return 200."""
        response = self._post(HIGH_RISK_POST)
        self.assertEqual(response.status_code, 200)

    def test_post_low_risk_returns_200(self):
        """Valid POST (low-risk profile) must return 200."""
        response = self._post(LOW_RISK_POST)
        self.assertEqual(response.status_code, 200)

    def test_result_page_contains_risk_classification(self):
        """Result page must show either 'Low Risk' or 'Elevated Risk'."""
        response = self._post(HIGH_RISK_POST)
        content = response.content.decode()
        has_low  = "Low Risk" in content
        has_high = "Elevated Risk" in content
        self.assertTrue(has_low or has_high,
                        "Result page must display a risk classification")

    def test_result_page_contains_probability_percent(self):
        """Result page must render the MLP probability as a percentage."""
        response = self._post(LOW_RISK_POST)
        self.assertIn(b"%", response.content,
                      "Probability percentage missing from result page")

    def test_model_cache_second_request_succeeds(self):
        """
        A second POST after the cache warms up must also return 200.
        Verifies that _model_cache is populated and reused without error.
        """
        self._post(HIGH_RISK_POST)   # warm the cache
        response = self._post(LOW_RISK_POST)
        self.assertEqual(response.status_code, 200)

    # --- POST with invalid data ---

    def test_post_missing_field_does_not_crash(self):
        """
        A POST missing a required field must return 200 (not 500)
        and include an error message in the rendered output.
        """
        bad_data = dict(HIGH_RISK_POST)
        del bad_data["age"]
        response = self._post(bad_data)
        self.assertEqual(response.status_code, 200,
                         "Missing field must not crash the server (500)")
        # The view renders form.html with context {"error": "Please fill in …"}
        # The template outputs the message text, not the variable name 'error'.
        self.assertIn(b"Please fill in all fields", response.content,
                      "Validation error message must be shown")

    def test_post_non_numeric_field_does_not_crash(self):
        """
        A POST with a non-numeric value for a numeric field must return 200
        and show a validation error instead of a TypeError traceback.
        """
        bad_data = dict(HIGH_RISK_POST)
        bad_data["age"] = "sixty-seven"
        response = self._post(bad_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Please fill in all fields", response.content)

    def test_post_extreme_numeric_values_do_not_crash(self):
        """
        Edge case: extreme but numeric values (age=150) must not raise
        an exception. Models accept any float input without bounds checks.
        """
        edge_data = dict(HIGH_RISK_POST)
        edge_data["age"] = 150
        response = self._post(edge_data)
        self.assertEqual(response.status_code, 200)
