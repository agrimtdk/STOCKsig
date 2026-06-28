import os
import pytest
import json
import time
import streamlit as st
from unittest.mock import MagicMock, patch
from src.dashboard.live_data import get_live_quote, load_favorites, save_favorites, toggle_favorite, FAVORITES_PATH, FAVORITES_DIR

@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    st.cache_data.clear()

def test_load_save_favorites_persistence(tmp_path):
    # Mock favorites file location using a temp directory path
    test_fav_path = tmp_path / "favorites.json"
    
    with patch("src.dashboard.live_data.FAVORITES_PATH", str(test_fav_path)), \
         patch("src.dashboard.live_data.FAVORITES_DIR", str(tmp_path)):
         
        # Verify initial is empty
        assert load_favorites() == []
        
        # Save some favorites
        favs = ["RELIANCE.NS", "TCS.NS"]
        save_favorites(favs)
        
        # Load and verify
        assert load_favorites() == favs

def test_toggle_favorites_logic(tmp_path):
    test_fav_path = tmp_path / "favorites.json"
    
    with patch("src.dashboard.live_data.FAVORITES_PATH", str(test_fav_path)), \
         patch("src.dashboard.live_data.FAVORITES_DIR", str(tmp_path)):
         
        assert load_favorites() == []
        
        # Toggle TCS on
        toggle_favorite("TCS.NS")
        assert load_favorites() == ["TCS.NS"]
        
        # Toggle RELIANCE on
        toggle_favorite("RELIANCE.NS")
        assert "RELIANCE.NS" in load_favorites()
        assert "TCS.NS" in load_favorites()
        
        # Toggle TCS off
        toggle_favorite("TCS.NS")
        assert load_favorites() == ["RELIANCE.NS"]

@patch("yfinance.Ticker")
def test_get_live_quote_with_fast_info(mock_ticker):
    # Mock fast_info dict values returned by yfinance Ticker
    mock_fast_info = {
        "last_price": 3000.0,
        "previous_close": 2950.0,
        "open": 2960.0,
        "day_high": 3010.0,
        "day_low": 2940.0,
        "last_volume": 1500000,
        "market_cap": 2000000000000.0
    }
    
    mock_t = MagicMock()
    mock_t.fast_info = mock_fast_info
    mock_ticker.return_value = mock_t
    
    quote = get_live_quote("RELIANCE.NS")
    
    assert quote["current_price"] == 3000.0
    assert quote["previous_close"] == 2950.0
    assert quote["open"] == 2960.0
    assert quote["high"] == 3010.0
    assert quote["low"] == 2940.0
    assert quote["volume"] == 1500000
    assert quote["day_change"] == 50.0
    assert quote["day_change_pct"] == (50.0 / 2950.0) * 100.0
    assert quote["market_cap"] == 2000000000000.0

@patch("yfinance.Ticker")
def test_get_live_quote_history_fallback(mock_ticker):
    # Mock history dataframe fallback when fast_info is empty/None
    import pandas as pd
    mock_hist = pd.DataFrame({
        "Open": [2960.0, 2980.0],
        "High": [2990.0, 3010.0],
        "Low": [2950.0, 2970.0],
        "Close": [2970.0, 3000.0],
        "Volume": [1000000, 1200000]
    }, index=[pd.Timestamp("2026-06-26"), pd.Timestamp("2026-06-27")])
    
    mock_t = MagicMock()
    mock_t.fast_info = None
    mock_t.history.return_value = mock_hist
    mock_ticker.return_value = mock_t
    
    quote = get_live_quote("RELIANCE.NS")
    
    assert quote["current_price"] == 3000.0
    assert quote["previous_close"] == 2970.0  # prior close from history row 1
    assert quote["high"] == 3010.0
    assert quote["low"] == 2970.0  # last Close is 3000, Low is 2970
    assert quote["volume"] == 1200000

def test_get_live_signal_for_ticker():
    from src.dashboard.signal_service import get_live_signal_for_ticker
    import pandas as pd
    import numpy as np
    
    # 1. Create a dummy MultiIndexed features DataFrame
    # Let's populate the columns required for technical indicator calculations
    dates = pd.date_range(start="2026-06-01", periods=30).strftime("%Y-%m-%d").tolist()
    data_dict = {
        "open": np.linspace(100.0, 120.0, 30),
        "high": np.linspace(102.0, 122.0, 30),
        "low": np.linspace(98.0, 118.0, 30),
        "close": np.linspace(101.0, 121.0, 30),
        "adj_close": np.linspace(101.0, 121.0, 30),
        "volume": np.linspace(100000, 150000, 30),
        "avg_news_sentiment": [0.1] * 30,
        "news_sentiment_std": [0.0] * 30,
        "news_count": [5] * 30,
        "transcript_sentiment": [0.2] * 30,
        "transcript_length": [1000] * 30,
        "positive_word_ratio": [0.05] * 30,
        "negative_word_ratio": [0.02] * 30,
        "macro_usdinr": [83.0] * 30,
        "macro_interest_rate": [6.5] * 30,
        "macro_cpi": [5.0] * 30,
    }
    
    # Generate index
    index = pd.MultiIndex.from_tuples(
        [("TCS.NS", d) for d in dates],
        names=["ticker", "date"]
    )
    df_feat = pd.DataFrame(data_dict, index=index)
    
    # Add indicator columns that compute_ticker_technicals usually creates (to satisfy fallback/checks)
    from src.features.technicals import compute_ticker_technicals
    # Technical indicator calculation needs date reset
    df_stock_reset = df_feat.reset_index()
    df_stock_res = compute_ticker_technicals(df_stock_reset)
    df_feat = df_stock_res.set_index(["ticker", "date"])
    
    # 2. Mock model and scaler
    mock_model = MagicMock()
    # predict_proba returns 2D array of class probabilities: [SELL, HOLD, BUY]
    mock_model.predict_proba.return_value = np.array([[0.1, 0.3, 0.6]])  # BUY signal
    
    payload = {
        "model": mock_model,
        "scaler": None
    }
    
    # Mock quote dict
    quote = {
        "open": 122.0,
        "high": 124.0,
        "low": 120.0,
        "current_price": 123.0,
        "volume": 160000,
        "day_change": 2.0,
        "day_change_pct": 1.6,
        "last_updated": "12:00:00"
    }
    
    # Mock json.load on feature_columns.json to return mock feature columns list
    feature_cols = ["open", "high", "low", "close", "volume", "sma_20", "rsi_14", "macd_hist", "volume_zscore_20"]
    
    from unittest.mock import mock_open
    with patch("builtins.open", mock_open(read_data='["open", "high", "low", "close", "volume", "sma_20", "rsi_14", "macd_hist", "volume_zscore_20"]')):
        with patch("json.load", return_value=feature_cols):
            res = get_live_signal_for_ticker("TCS.NS", quote, df_feat, payload)
            
            assert res is not None
            assert res["label"] in ["BUY", "SELL", "HOLD"]
            assert "prob_buy" in res
            assert "df_stock_live" in res
            assert "sig_live_df" in res

def test_filter_df_by_period():
    from app import filter_df_by_period
    import pandas as pd
    
    # Create DataFrame with daily dates over 2 years
    dates = pd.date_range(start="2024-01-01", end="2026-01-01", freq="D").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame({"close": range(len(dates))}, index=pd.Index(dates, name="date"))
    
    # Verify filter bounds
    df_1m = filter_df_by_period(df, "1M")
    assert not df_1m.empty
    # Length of 1 Month is roughly 30-31 days
    assert 28 <= len(df_1m) <= 32
    
    df_1y = filter_df_by_period(df, "1Y")
    assert 364 <= len(df_1y) <= 367
    
    df_max = filter_df_by_period(df, "MAX")
    assert len(df_max) == len(df)


