"""Zadanie 1: wczytywanie i podział danych (zbiór wine z lab. 2).

Funkcje load_data i split_data nie mają efektów ubocznych (żadnych printów ani zapisów
do plików), więc można je bezpiecznie importować w train.py, evaluate.py i w testach.
"""
import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

# Zadanie 4: konfiguracja z pliku .env (wartości domyślne na wypadek braku klucza)
load_dotenv()
DATA_PATH = os.getenv("DATA_PATH", "data/wine.csv")
TARGET_COLUMN = os.getenv("TARGET_COLUMN", "target")
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", 42))


def load_data(path=DATA_PATH):
    """Wczytuje dane z pliku CSV i zwraca DataFrame.

    Brak pliku zgłasza FileNotFoundError (zachowanie pandas), a pusty plik ValueError.
    """
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Plik {path} jest pusty") from e


def split_data(df, test_size=TEST_SIZE, seed=RANDOM_SEED, target=TARGET_COLUMN):
    """Dzieli DataFrame na (X_train, X_test, y_train, y_test) z zachowaniem proporcji klas."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df musi być obiektem pandas.DataFrame")
    if df.empty:
        raise ValueError("Zbiór danych jest pusty")
    if target not in df.columns:
        raise ValueError(f"Brak kolumny docelowej '{target}' w danych")
    if not 0 < test_size < 1:
        raise ValueError("test_size musi należeć do przedziału (0, 1)")

    X = df.drop(columns=target)
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


if __name__ == "__main__":
    # Uruchomienie bezpośrednie: jednorazowo zapisuje zbiór wine do CSV pod DATA_PATH
    # i pokazuje rozmiary podziału. Etap "data" w pipeline CI.
    from sklearn.datasets import load_wine

    if not os.path.exists(DATA_PATH):
        os.makedirs(os.path.dirname(DATA_PATH) or ".", exist_ok=True)
        load_wine(as_frame=True).frame.to_csv(DATA_PATH, index=False)
        print(f"Zapisano zbiór wine do {DATA_PATH}")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Dane: {df.shape}, trening: {X_train.shape}, test: {X_test.shape}")
