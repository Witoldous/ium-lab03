# IUM – Lab 3: strukturyzacja pipeline'u pod CI

Pipeline z lab. 2 (zbiór wine, Pipeline `StandardScaler` + `LogisticRegression`) rozbity na niezależne moduły.

```
src/data_prep.py   # wczytanie i podział danych (python src/data_prep.py zapisuje data/wine.csv)
src/train.py       # trening i zapis modelu do models/model.joblib
src/evaluate.py    # ewaluacja zapisanego modelu, wynik w wyniki.txt
tests/             # testy pytest dla data_prep.py
.env.example       # klucze konfiguracji (skopiuj do .env i uzupełnij)
```

## Uruchomienie (z katalogu głównego projektu)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # i uzupełnij wartości
python src/data_prep.py
python src/train.py
python src/evaluate.py
pytest
```
