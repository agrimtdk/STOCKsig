import os
import pytest
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path

from src.utils.config import config
from src.signals.threshold_optimizer import optimize_thresholds
from src.signals.signal_generator import generate_signals
from src.signals.filters import check_buy_confirmations, check_sell_confirmations

class DummyModel:
    def predict_proba(self, X):
        # Row 1: High BUY prob (0.1, 0.2, 0.7) -> passes thresholds, passes buy filter
        # Row 2: High BUY prob (0.1, 0.1, 0.8) -> passes thresholds, passes buy filter (RSI is 60 < 75)
        # Row 3: High SELL prob (0.6, 0.3, 0.1) -> passes thresholds, fails sell filter (RSI is 25 <= 25)
        # Row 4: High BUY prob (0.1, 0.1, 0.8) -> passes thresholds, fails buy filter (RSI is 80 >= 75)
        # Row 5: High SELL prob (0.7, 0.2, 0.1) -> passes thresholds, passes sell filter (close < sma, RSI=45 > 25, macd_hist=-1.0 < 0.05)
        # Row 6: Low probs (0.3, 0.4, 0.3) -> HOLD
        return np.array([
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.3, 0.4, 0.3]
        ])

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    """Overrides config paths to use a temporary sandbox directory for all signal tests."""
    # Monkeypatch config properties on the Config class level
    from src.utils.config import Config
    monkeypatch.setattr(Config, "project_root", property(lambda self: tmp_path))
    monkeypatch.setattr(Config, "tickers", property(lambda self: ["TCS.NS"]))
    
    # Create necessary folders inside the sandbox
    (tmp_path / "data" / "raw" / "prices").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "predictions").mkdir(parents=True, exist_ok=True)
    
    test_features_dir = tmp_path / "data" / "features"
    test_features_dir.mkdir(parents=True, exist_ok=True)
    test_features_file = test_features_dir / "features.parquet"
    
    # Seed mock features (6 rows with various technical parameters)
    df = pd.DataFrame({
        "close": [100.0, 105.0, 95.0, 100.0, 95.0, 95.0],
        "sma_20": [98.0, 98.0, 98.0, 102.0, 102.0, 102.0],
        "rsi_14": [45.0, 60.0, 25.0, 80.0, 45.0, 30.0],
        "macd_hist": [1.0, 0.5, -0.5, 1.5, -1.0, -2.0],
        "volume_zscore_20": [0.5, 0.8, -1.2, 1.2, -0.5, -1.5],
        # Required to run preprocessor list loader check
        "sma_5": [100.0]*6,
        "macd": [0.0]*6,
        "macd_signal": [0.0]*6,
        "avg_news_sentiment": [0.0]*6,
        "transcript_sentiment": [0.0]*6,
        "macro_usdinr": [83.0]*6,
    }, index=pd.MultiIndex.from_product([["TCS.NS"], [f"2025-01-0{i}" for i in range(1, 7)]], names=['ticker', 'date']))
    df.to_parquet(test_features_file)
    
    # Seed reports/feature_columns.json
    feature_cols = [
        "close", "sma_20", "rsi_14", "macd_hist", "volume_zscore_20", 
        "sma_5", "macd", "macd_signal", "avg_news_sentiment", 
        "transcript_sentiment", "macro_usdinr"
    ]
    with open(tmp_path / "reports" / "feature_columns.json", 'w') as f:
        json.dump(feature_cols, f)
        
    # Seed mock calibrated model pickle
    mock_payload = {
        'model': DummyModel(),
        'scaler': None
    }
    with open(tmp_path / "models" / "calibrated_best.pkl", 'wb') as f:
        pickle.dump(mock_payload, f)
        
    # Seed predictions/oof_predictions.parquet (needed for threshold optimizer test)
    # 20 rows of predictions to satisfy signal frequency check
    dates_oof = [f"2025-01-{i:02d}" for i in range(1, 21)]
    oof_df = pd.DataFrame({
        "true_label": np.random.choice([0, 1, 2], 20),
        "pred_label": np.random.choice([0, 1, 2], 20),
        "prob_sell": np.array([0.1]*14 + [0.65]*6),
        "prob_hold": np.array([0.25]*20),
        "prob_buy": np.array([0.65]*6 + [0.1]*14),
    }, index=pd.MultiIndex.from_product([["TCS.NS"], dates_oof], names=['ticker', 'date']))
    oof_df.to_parquet(tmp_path / "data" / "predictions" / "oof_predictions.parquet")
    
    return test_features_file

def test_filters(setup_test_env):
    """Validates that technical filter confirmations correctly evaluate conditions."""
    df = pd.read_parquet(setup_test_env)
    
    buy_confirm = check_buy_confirmations(df)
    sell_confirm = check_sell_confirmations(df)
    
    # Row 1: close=100 > sma_20*0.98, rsi=45 < 75, macd_hist=1.0 > -0.05, vol_z=0.5 > -1 -> True
    assert buy_confirm.iloc[0] == True
    
    # Row 3: close=95 < sma_20=98, rsi=25 <= 25 -> False for SELL (needs RSI > 25)
    assert sell_confirm.iloc[2] == False
    
    # Row 4: close=100 > sma_20=102*0.98, rsi=80 >= 75 -> False for BUY
    assert buy_confirm.iloc[3] == False
    
    # Row 5: close=95 < sma_20=102*1.02, rsi=45 > 25, macd_hist=-1.0 < 0.05 -> True for SELL
    assert sell_confirm.iloc[4] == True
    
    # Row 2: close=105 > sma_20=98*1.02 -> False for SELL (needs close < sma_20 * 1.02)
    assert sell_confirm.iloc[1] == False

def test_threshold_optimizer(setup_test_env):
    """Validates that threshold optimizer runs and finds valid thresholds."""
    buy_thresh, sell_thresh = optimize_thresholds()
    
    # Thresholds should be floats in the grid [0.35, 0.70]
    assert 0.35 <= buy_thresh <= 0.70
    assert 0.35 <= sell_thresh <= 0.70
    
    # JSON report should exist
    report_path = config.project_root / "reports" / "threshold_report.json"
    assert report_path.exists()

def test_signal_generation_and_conversions(setup_test_env, tmp_path):
    """Validates signal generation, filter rejections, confidence bounds, and file save."""
    # Use thresholds that trigger trades on our dummy model probabilities: buy=0.60, sell=0.50
    signals_df, stats = generate_signals(
        buy_threshold=0.60,
        sell_threshold=0.50
    )
    
    # 1. Output Parquet saved
    signals_parquet = tmp_path / "data" / "signals" / "signals.parquet"
    signals_parquet.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(signals_parquet)
    assert signals_parquet.exists()
    
    # 2. Check signal columns
    expected_cols = {'signal', 'confidence_score', 'prob_sell', 'prob_hold', 'prob_buy'}
    assert expected_cols.issubset(signals_df.columns)
    
    # Reset index for testing rows
    df_reset = signals_df.reset_index()
    
    # 3. Check signal boundaries
    # Row 1: prob_buy = 0.7 > 0.60, confirmed by filters -> BUY (1)
    assert df_reset.loc[0, 'signal'] == 1
    
    # Row 2: prob_buy = 0.8 > 0.60, confirmed by filters -> BUY (1)
    assert df_reset.loc[1, 'signal'] == 1
    
    # Row 3: prob_sell = 0.6 > 0.50, rejected by filters (RSI=25 <= 25) -> HOLD (0)
    assert df_reset.loc[2, 'signal'] == 0
    assert stats['rejections']['SELL_rejected'] >= 1
    
    # Row 4: prob_buy = 0.8 > 0.60, rejected by filters (RSI=80 >= 75) -> HOLD (0)
    assert df_reset.loc[3, 'signal'] == 0
    assert stats['rejections']['BUY_rejected'] >= 1
    
    # Row 5: prob_sell = 0.7 > 0.50, confirmed by filters -> SELL (-1)
    assert df_reset.loc[4, 'signal'] == -1
    
    # Row 6: prob_hold = 0.4 (probs: 0.3, 0.4, 0.3), doesn't pass -> HOLD (0)
    assert df_reset.loc[5, 'signal'] == 0
    
    # 4. Check confidence limits (0-100)
    assert (df_reset['confidence_score'] >= 0.0).all()
    assert (df_reset['confidence_score'] <= 100.0).all()
    
    # Verify exact confidence values
    # For Row 1 (BUY): prob_buy = 0.7 * 100 = 70.0
    assert np.isclose(df_reset.loc[0, 'confidence_score'], 70.0)
    
    # For Row 3 (Downgraded to HOLD): prob_hold = 0.3 * 100 = 30.0
    assert np.isclose(df_reset.loc[2, 'confidence_score'], 30.0)

def test_percentile_fallback(setup_test_env):
    """Validates that percentile fallback triggers when final confirmed signal count is 0."""
    # Run with extremely high thresholds that yield 0 signals: buy=0.99, sell=0.99
    signals_df, stats = generate_signals(
        buy_threshold=0.99,
        sell_threshold=0.99
    )
    
    # Fallback should be triggered
    assert stats['fallback_triggered']['BUY'] == True
    assert stats['fallback_triggered']['SELL'] == True
    
    # Final thresholds should be overridden to OOF 75th percentile (0.65 in our mock data)
    assert stats['final_buy_threshold'] == 0.65
    assert stats['final_sell_threshold'] == 0.65
