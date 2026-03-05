import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_heart_attack_data():
    """Load the heart dataset and return feature matrix X and labels y."""
    csv_path = os.path.join(SCRIPT_DIR, 'heart.csv')
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = SVC(probability=True)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Return accuracy score. No GUI side-effects."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)