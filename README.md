# Surge Heartly 🫀

An AI-powered **Heart Attack Risk Assessment** web application built with Django. It uses an ensemble of four machine learning models to predict a patient's cardiac risk from 13 clinical features.

---

## ✨ Features

- **4-Model Ensemble** — Logistic Regression, Random Forest, SVM, and a Multi-Layer Perceptron vote together for a robust final prediction.
- **Model Caching** — All models are trained once on first run and cached in memory, so subsequent predictions are near-instant.
- **Modern UI** — Clean "Medical Tech" aesthetic with Google Fonts, glassmorphism cards, animated progress bars, and per-model result badges.
- **Input Validation** — User-friendly error messages on invalid form data (no raw 500 errors).

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Surge_Heartly.git
cd Surge_Heartly
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run database migrations
```bash
python manage.py migrate
```

### 5. Start the server
```bash
python manage.py runserver
```

Open your browser at **http://127.0.0.1:8000/**

> **First request:** The four ML models are trained on startup — this takes ~10–30 seconds. Every subsequent prediction is cached and returns instantly.

---

## 🧠 Models Used

| Model | Library |
|---|---|
| Logistic Regression | scikit-learn |
| Random Forest | scikit-learn |
| Support Vector Machine | scikit-learn |
| Multi-Layer Perceptron (Neural Network) | scikit-learn |

The final prediction is determined by a **majority vote** (≥ 2 of 4 models). The Neural Network also provides an individual probability percentage displayed as a progress bar.

---

## 📋 Input Features

The app accepts the following 13 clinical features from the [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease):

| Feature | Description |
|---|---|
| Age | Patient's age in years |
| Sex | Biological sex (0 = Female, 1 = Male) |
| CP | Chest pain type (0–3) |
| Trestbps | Resting blood pressure (mm Hg) |
| Chol | Serum cholesterol (mg/dl) |
| FBS | Fasting blood sugar > 120 mg/dl |
| Restecg | Resting ECG results (0–2) |
| Thalach | Maximum heart rate achieved |
| Exang | Exercise induced angina |
| Oldpeak | ST depression (exercise vs rest) |
| Slope | Slope of peak exercise ST segment |
| CA | Major vessels colored by fluoroscopy (0–4) |
| Thal | Thalassemia result (0–2) |

---

## 🗂 Project Structure

```
Surge_Heartly/
├── main/
│   ├── templates/
│   │   ├── base.html         # Global layout, fonts, CSS
│   │   ├── form.html         # Patient input form
│   │   └── result.html       # Prediction results page
│   ├── heart.csv             # UCI Heart Disease dataset
│   ├── models.py
│   ├── neural_network.py     # MLP model training & inference
│   ├── prediction_system.py  # LR / RF / SVM training & evaluation
│   ├── urls.py
│   └── views.py              # Request handling + model cache
├── surge_heartly/
│   ├── settings.py
│   └── urls.py
├── manage.py
└── requirements.txt
```

---

## ⚙️ Requirements

- Python 3.9+
- See `requirements.txt` for full dependency list

---

## 📄 License

MIT
