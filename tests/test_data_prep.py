"""Zadanie 5: testy jednostkowe funkcji z data_prep.py."""
import pandas as pd
import pytest
from sklearn.datasets import load_wine

from data_prep import load_data, split_data


@pytest.fixture
def df():
    """Zbiór wine jako DataFrame - niezależny od pliku CSV i od .env."""
    return load_wine(as_frame=True).frame


def test_split_sizes(df):
    """Zbiór testowy ma rozmiar zgodny z proporcją test_size."""
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, seed=42)
    assert len(X_test) == pytest.approx(len(df) * 0.2, abs=1)
    assert len(X_test) == len(y_test)
    assert len(X_train) == len(y_train)


def test_split_sum_equals_total(df):
    """Trening + test = cały zbiór, bez zgubionych i zdublowanych próbek."""
    X_train, X_test, _, _ = split_data(df, test_size=0.25, seed=0)
    assert len(X_train) + len(X_test) == len(df)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_removes_target_from_features(df):
    """Kolumna docelowa trafia tylko do y, nie do X."""
    X_train, X_test, y_train, _ = split_data(df)
    assert "target" not in X_train.columns
    assert "target" not in X_test.columns
    assert y_train.name == "target"


def test_split_is_reproducible(df):
    """To samo ziarno daje ten sam podział."""
    a = split_data(df, seed=42)[1]
    b = split_data(df, seed=42)[1]
    assert a.index.equals(b.index)


def test_split_empty_dataframe_raises():
    with pytest.raises(ValueError, match="pusty"):
        split_data(pd.DataFrame(columns=["x", "target"]))


def test_split_missing_target_raises(df):
    with pytest.raises(ValueError, match="kolumny docelowej"):
        split_data(df.drop(columns="target"))


@pytest.mark.parametrize("test_size", [0, 1, -0.2, 1.5])
def test_split_invalid_test_size_raises(df, test_size):
    with pytest.raises(ValueError, match="test_size"):
        split_data(df, test_size=test_size)


def test_split_wrong_type_raises():
    with pytest.raises(TypeError):
        split_data([[1, 2], [3, 4]])


def test_load_data_returns_dataframe(df, tmp_path):
    path = tmp_path / "wine.csv"
    df.to_csv(path, index=False)
    loaded = load_data(path)
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == df.shape


def test_load_data_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "brak.csv")


def test_load_data_empty_file_raises(tmp_path):
    path = tmp_path / "pusty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="pusty"):
        load_data(path)
