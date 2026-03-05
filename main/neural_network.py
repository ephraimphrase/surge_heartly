import os
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Resolve paths relative to this file so they work regardless of CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(SCRIPT_DIR, 'heart_attack_scaler.pkl')
MODEL_PATH  = os.path.join(SCRIPT_DIR, 'heart_attack_mlp_model.pkl')


def train_neural_network(X, y):
    """
    Train a multi-layer perceptron (MLP) on the full dataset X/y.
    Internally splits off a validation set, fits a scaler, saves both
    the scaler and the trained model to disk, and returns the model.

    Uses scikit-learn MLPClassifier (architecture mirrors the original
    Keras network: 13 → 16 → 8 → 1) for broad Python version support.
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Scale features and persist the scaler for use at inference time
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)

    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),   # mirrors Dense(16) → Dense(8) → output
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        verbose=False,
    )
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, MODEL_PATH)
    return model


def predict(model, age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal):
    """
    Scale a single patient record and return (binary_result, probability_pct).
    """
    input_data = np.array(
        [[age, sex, cp, trestbps, chol, fbs, restecg,
          thalach, exang, oldpeak, slope, ca, thal]],
        dtype=float
    )

    scaler = joblib.load(SCALER_PATH)
    input_data = scaler.transform(input_data)

    raw = float(model.predict_proba(input_data)[0][1])  # P(class=1)
    probability = round(raw * 100, 2)
    result = 1 if raw >= 0.5 else 0

    return result, probability


def evaluate_nn(model, X_test, y_test):
    """Return accuracy score (no GUI side-effects)."""
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    return accuracy_score(y_test, y_pred)
