"""
Tests for the DataLoader class.
"""

from pathlib import Path

import pandas as pd
import pytest

from ml_trading_system.data.data_loader import DataLoader


def test_data_loader_init(tmp_path: Path):
    loader = DataLoader(data_dir=tmp_path)
    assert loader.data_dir.exists()


def test_save_and_load(tmp_path: Path):
    loader = DataLoader(data_dir=tmp_path)
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0]}, index=pd.date_range("2023-01-01", periods=3)
    )
    loader.save(df, "test.csv")
    loaded = loader.load("test.csv")
    assert len(loaded) == 3
    assert "Close" in loaded.columns


def test_load_missing_file_raises(tmp_path: Path):
    loader = DataLoader(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent.csv")
