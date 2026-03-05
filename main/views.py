import logging
import numpy as np
from django.shortcuts import render

from .prediction_system import (
    load_heart_attack_data,
    train_logistic_regression,
    train_random_forest,
    train_svm,
)
from .neural_network import train_neural_network, predict as nn_predict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level model cache — models are trained once on the first request
# and then reused for every subsequent request, making prediction fast.
# ---------------------------------------------------------------------------
_model_cache = {}


def _get_models():
    """Return cached models, training them on first call."""
    if not _model_cache:
        logger.info("Training ML models for the first time…")
        X, y = load_heart_attack_data()
        _model_cache['X'] = X
        _model_cache['y'] = y
        _model_cache['lr'] = train_logistic_regression(X, y)
        _model_cache['rf'] = train_random_forest(X, y)
        _model_cache['svm'] = train_svm(X, y)
        _model_cache['nn'] = train_neural_network(X, y)
        logger.info("Model training complete.")
    return _model_cache


def heart_attack_prediction(request):
    if request.method != "POST":
        return render(request, "form.html")

    # -----------------------------------------------------------------------
    # Input parsing with validation
    # -----------------------------------------------------------------------
    try:
        age      = int(request.POST.get("age", ""))
        sex      = int(request.POST.get("sex", ""))
        cp       = int(request.POST.get("cp", ""))
        trestbps = int(request.POST.get("trestbps", ""))
        chol     = int(request.POST.get("chol", ""))
        fbs      = int(request.POST.get("fbs", ""))
        restecg  = int(request.POST.get("restecg", ""))
        thalach  = int(request.POST.get("thalach", ""))
        exang    = int(request.POST.get("exang", ""))
        oldpeak  = float(request.POST.get("oldpeak", ""))
        slope    = int(request.POST.get("slope", ""))
        ca       = int(request.POST.get("ca", ""))
        thal     = int(request.POST.get("thal", ""))
    except (ValueError, TypeError):
        return render(request, "form.html", {
            "error": "Please fill in all fields with valid numeric values."
        })

    # -----------------------------------------------------------------------
    # Load cached models and run predictions
    # -----------------------------------------------------------------------
    try:
        models = _get_models()

        input_data = np.array(
            [[age, sex, cp, trestbps, chol, fbs, restecg,
              thalach, exang, oldpeak, slope, ca, thal]]
        )

        lr_pred  = int(models['lr'].predict(input_data)[0])
        rf_pred  = int(models['rf'].predict(input_data)[0])
        svm_pred = int(models['svm'].predict(input_data)[0])
        nn_pred, nn_probability = nn_predict(
            models['nn'],
            age, sex, cp, trestbps, chol, fbs,
            restecg, thalach, exang, oldpeak, slope, ca, thal,
        )

        # Majority vote across all four models for a robust final result
        votes = lr_pred + rf_pred + svm_pred + nn_pred
        majority_prediction = 1 if votes >= 2 else 0

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        return render(request, "form.html", {
            "error": "An error occurred during prediction. Please try again."
        })

    context = {
        "majority_prediction": majority_prediction,
        "nn_probability": nn_probability,
        "lr_pred":  lr_pred,
        "rf_pred":  rf_pred,
        "svm_pred": svm_pred,
        "nn_pred":  nn_pred,
    }
    return render(request, "result.html", context)
