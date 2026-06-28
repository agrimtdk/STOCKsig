import os
import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from src.utils.config import config
from src.models.preprocessing import load_and_preprocess_data
from src.models.training import train_walk_forward
from src.models.evaluation import generate_evaluation_reports
from src.models.registry import save_and_calibrate_models

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    """Overrides config paths to use a temporary sandbox directory for all training tests."""
    # Monkeypatch config properties on the Config class level
    from src.utils.config import Config
    monkeypatch.setattr(Config, "project_root", property(lambda self: tmp_path))
    monkeypatch.setattr(Config, "tickers", property(lambda self: ["TCS.NS", "INFY.NS"]))
    
    # Create the data and reports folders inside the sandbox
    (tmp_path / "data" / "raw" / "prices").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    
    test_features_dir = tmp_path / "data" / "features"
    test_features_dir.mkdir(parents=True, exist_ok=True)
    test_features_file = test_features_dir / "features.parquet"
    
    # Seed mock features Parquet file (100 rows per ticker, 2 tickers)
    dates = pd.date_range(start="2025-01-01", end="2025-04-10", freq="D")
    n_days = len(dates)
    
    # Simple values
    close = np.linspace(100.0, 150.0, n_days)
    labels = np.random.choice([-1, 0, 1], n_days)
    
    # Create two dataframes
    df1 = pd.DataFrame({
        "close": close,
        "sma_5": close - 1.0,
        "rsi_14": np.linspace(40, 60, n_days),
        "macd": np.random.normal(0, 0.1, n_days),
        "avg_news_sentiment": np.random.normal(0, 0.2, n_days),
        "transcript_sentiment": np.random.normal(0, 0.3, n_days),
        "macro_usdinr": np.linspace(82.0, 83.5, n_days),
        "future_return_5d": np.random.normal(0.01, 0.02, n_days),
        "label": labels
    }, index=pd.MultiIndex.from_product([["TCS.NS"], dates.strftime('%Y-%m-%d')], names=['ticker', 'date']))
    
    df2 = pd.DataFrame({
        "close": close * 0.5,
        "sma_5": close * 0.5 - 0.5,
        "rsi_14": np.linspace(35, 55, n_days),
        "macd": np.random.normal(0, 0.1, n_days),
        "avg_news_sentiment": np.random.normal(0, 0.2, n_days),
        "transcript_sentiment": np.random.normal(0, 0.3, n_days),
        "macro_usdinr": np.linspace(82.0, 83.5, n_days),
        "future_return_5d": np.random.normal(-0.01, 0.02, n_days),
        "label": labels
    }, index=pd.MultiIndex.from_product([["INFY.NS"], dates.strftime('%Y-%m-%d')], names=['ticker', 'date']))
    
    mock_features = pd.concat([df1, df2])
    # Save Parquet
    mock_features.to_parquet(test_features_file)
    
    return test_features_file

def test_preprocessing_and_mapping(setup_test_env, tmp_path):
    """Validates loading, no NaN targets verification, and integer mapping checks."""
    X, y, feature_cols = load_and_preprocess_data(setup_test_env)
    
    # Verify X indices match y indices
    assert (X.index == y.index).all()
    
    # Verify no NaNs in target
    assert y.notna().all()
    
    # Verify integer mapping: SELL (-1) -> 0, HOLD (0) -> 1, BUY (1) -> 2
    assert set(y.unique()).issubset({0, 1, 2})
    
    # Verify metadata columns are removed
    assert "close" not in X.columns
    assert "future_return_5d" not in X.columns
    assert "label" not in X.columns
    
    # Verify JSON list export
    json_path = tmp_path / "reports" / "feature_columns.json"
    assert json_path.exists()
    
    with open(json_path, 'r') as f:
        stored_features = json.load(f)
    assert stored_features == feature_cols

def test_chronological_timeseriessplit_no_shuffling(setup_test_env):
    """Validates that TimeSeriesSplit preserves chronological date sorting and doesn't shuffle."""
    X, y, _ = load_and_preprocess_data(setup_test_env)
    
    # Re-sort to match training structure
    df_temp = X.copy()
    df_temp['label'] = y
    df_temp = df_temp.reset_index()
    df_temp = df_temp.sort_values(by=['date', 'ticker']).reset_index(drop=True)
    
    X_sorted = df_temp.drop(columns=['ticker', 'date', 'label'])
    y_sorted = df_temp['label']
    dates_sorted = pd.to_datetime(df_temp['date'])
    
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_sorted)):
        train_dates = dates_sorted.iloc[train_idx]
        val_dates = dates_sorted.iloc[val_idx]
        
        # Verify train index elements are before validation index elements chronologically
        assert train_dates.max() <= val_dates.min()
        
        # Verify indices are contiguous sequences (no shuffling)
        assert np.array_equal(train_idx, np.arange(len(train_idx)))
        assert np.array_equal(val_idx, np.arange(len(train_idx), len(train_idx) + len(val_idx)))

def test_training_pipeline_and_oof_predictions(setup_test_env, tmp_path):
    """Validates probability output shapes, OOF saves, and pickle files validation."""
    X, y, feature_cols = load_and_preprocess_data(setup_test_env)
    
    # Run training
    results = train_walk_forward(X, y)
    
    # Verify 3-class probability shapes (prob_sell, prob_hold, prob_buy)
    for model_name, oof_df in results['oof_predictions'].items():
        assert 'prob_sell' in oof_df.columns
        assert 'prob_hold' in oof_df.columns
        assert 'prob_buy' in oof_df.columns
        assert 'pred_label' in oof_df.columns
        assert 'true_label' in oof_df.columns
        
        # Probabilities sum to 1.0 (approx)
        prob_sum = oof_df['prob_sell'] + oof_df['prob_hold'] + oof_df['prob_buy']
        assert np.allclose(prob_sum, 1.0, atol=1e-5)
        
        # Out-of-fold prediction count should be exactly the size of all validation splits.
        # In TimeSeriesSplit(n_splits=5) for N rows, each split size = N / 6.
        # Total validation size = 5 * (N/6). For N = 200, val size = 5 * 33 = 165.
        assert len(oof_df) == 165
        
    # Run evaluations & registry
    metrics = generate_evaluation_reports(results, X, y, feature_cols)
    best_model = save_and_calibrate_models(results, X, y)
    
    # Verify best model and calibrated best model exist in registry
    models_dir = tmp_path / "models"
    assert (models_dir / f"{best_model}.pkl").exists()
    assert (models_dir / "best_model.pkl").exists()
    assert (models_dir / "calibrated_best.pkl").exists()
    
    # Verify calibrated best loaded successfully and matches prefix cv structure
    with open(models_dir / "calibrated_best.pkl", 'rb') as f:
        payload = pickle.load(f)
    assert 'model' in payload
    assert payload['model'].cv == 5
    
    # Verify feature importance csv generated correctly
    importance_path = tmp_path / "reports" / "feature_importance.csv"
    assert importance_path.exists()
    
    feat_df = pd.read_csv(importance_path)
    assert 'xgboost_importance' in feat_df.columns
    assert 'lightgbm_importance' in feat_df.columns
    assert 'mean_importance' in feat_df.columns
    assert len(feat_df) == len(feature_cols)
    
    # Check that sorting is descending by mean importance
    assert feat_df['mean_importance'].is_monotonic_decreasing
