"""Zadanie 3: ewaluacja zapisanego modelu na zbiorze testowym (bez ponownego treningu)."""
import os

import joblib
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, classification_report

from data_prep import DATA_PATH, load_data, split_data

# Zadanie 4: ścieżki artefaktów z pliku .env
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
RESULTS_PATH = os.getenv("RESULTS_PATH", "wyniki.txt")


def evaluate(model, X_test, y_test):
    """Zwraca accuracy i raport klasyfikacji dla danych testowych."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred), classification_report(y_test, y_pred, digits=4)


if __name__ == "__main__":
    model = joblib.load(MODEL_PATH)             # ten sam format co w train.py
    # Ten sam podział (to samo ziarno z .env) co przy treningu -> model nie widział tych danych
    _, X_test, _, y_test = split_data(load_data(DATA_PATH))
    acc, report = evaluate(model, X_test, y_test)

    text = f"Model: {MODEL_PATH}\nDane testowe: {len(X_test)} próbek\nAccuracy: {acc:.4f}\n\n{report}"
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Zapisano wyniki do {RESULTS_PATH}")
