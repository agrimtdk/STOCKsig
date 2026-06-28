import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from src.dashboard.data_service import (
    load_features, load_signals, load_equity_curve, load_trade_log,
    load_feature_importance, load_backtest_report, load_benchmark_comparison
)
from src.dashboard.signal_service import (
    get_latest_signal_info, get_technical_snapshot, get_sentiment_snapshot,
    get_market_regime, get_signal_explainability
)
from src.dashboard.metrics_service import get_volatility_risk_level, format_metric_value
from src.dashboard.charts import (
    create_candlestick_chart, create_equity_curve_chart,
    create_feature_importance_chart, create_sparkline,
    create_probability_histograms, create_confidence_gauge,
    create_signal_history_timeline, create_momentum_charts
)
from app import get_roundtrip_trades

# Create mock dataframes for testing service layers
@pytest.fixture
def mock_features():
    idx = pd.MultiIndex.from_product([["ASIANPAINT.NS"], ["2026-01-01", "2026-01-02"]], names=["ticker", "date"])
    df = pd.DataFrame(index=idx)
    df["open"] = 100.0
    df["close"] = 105.0
    df["high"] = 110.0
    df["low"] = 95.0
    df["adj_close"] = 105.0
    
    # Non-constant values to ensure non-zero std and variance
    df["rsi_14"] = [30.0, 60.0]
    df["macd_hist"] = [0.01, 0.08]
    df["atr_14"] = [2.0, 3.0]
    df["bollinger_width"] = [0.10, 0.20]
    df["daily_volatility_20d"] = [0.01, 0.02]
    df["avg_news_sentiment"] = [0.10, 0.30]
    df["news_count"] = [3, 7]
    df["transcript_sentiment"] = [0.20, 0.40]
    
    df["sma_20"] = 100.0
    df["ema_20"] = 101.0
    return df

@pytest.fixture
def mock_signals():
    idx = pd.MultiIndex.from_product([["ASIANPAINT.NS"], ["2026-01-01", "2026-01-02"]], names=["ticker", "date"])
    df = pd.DataFrame(index=idx)
    df["signal"] = 1
    df["confidence_score"] = 75.0
    df["prob_sell"] = 0.10
    df["prob_hold"] = 0.15
    df["prob_buy"] = 0.75
    return df

@pytest.fixture
def mock_trade_log():
    return pd.DataFrame([
        {"ticker": "ASIANPAINT.NS", "action": "BUY", "date": "2026-01-01", "price": 100.0, "shares": 10, "realized_pnl": 0.0},
        {"ticker": "ASIANPAINT.NS", "action": "SELL", "date": "2026-01-02", "price": 105.0, "shares": 10, "realized_pnl": 50.0}
    ])

# 1. TEST DATA SERVICE READS (Real Files)
def test_real_files_load_and_data_caches():
    """Validates that real files are readable, load correctly, or return empty DataFrames without crashing."""
    df_feat = load_features()
    df_sig = load_signals()
    df_equity = load_equity_curve()
    df_trades = load_trade_log()
    df_imp = load_feature_importance()
    report = load_backtest_report()
    bench = load_benchmark_comparison()
    
    assert isinstance(df_feat, pd.DataFrame)
    assert isinstance(df_sig, pd.DataFrame)
    assert isinstance(df_equity, pd.DataFrame)
    assert isinstance(df_trades, pd.DataFrame)
    assert isinstance(df_imp, pd.DataFrame)
    assert isinstance(report, dict)
    assert isinstance(bench, dict)

# 2. TEST SIGNAL SERVICE LOOKUPS
def test_signal_service_lookups(mock_features, mock_signals):
    # Test Signal Info Lookup
    sig_info = get_latest_signal_info(mock_signals, "ASIANPAINT.NS", "2026-01-02")
    assert sig_info["ticker"] == "ASIANPAINT.NS"
    assert sig_info["label"] == "BUY"
    assert sig_info["confidence_score"] == 75.0
    assert sig_info["prob_buy"] == 0.75
    
    # Test Technical Snapshot
    tech = get_technical_snapshot(mock_features, "ASIANPAINT.NS", "2026-01-02")
    assert tech["rsi"] == 60.0
    assert tech["volatility"] == 2.0
    
    # Test Sentiment Snapshot
    sent = get_sentiment_snapshot(mock_features, "ASIANPAINT.NS", "2026-01-02")
    assert sent["news_sentiment"] == 0.30
    assert sent["news_count"] == 7
    assert sent["transcript_sentiment"] == 0.40

# 3. TEST MARKET REGIME AND EXPLAINABILITY
def test_market_regime_and_explainability(mock_features):
    # Test Market Regime
    regime = get_market_regime(mock_features, "ASIANPAINT.NS", "2026-01-02")
    assert regime["trend"] == "Bullish"
    assert regime["volatility"] == "Medium"
    assert regime["sentiment"] == "Positive"
    
    # Test Explainability (Mock)
    feat_imp = pd.DataFrame([
        {"feature": "rsi_14", "mean_importance": 0.10},
        {"feature": "macd_hist", "mean_importance": 0.08}
    ])
    explain = get_signal_explainability(mock_features, "ASIANPAINT.NS", "2026-01-02", feat_imp, 1)
    assert len(explain) > 0
    assert explain[0]["feature"] in ["rsi_14", "macd_hist"]

# 4. TEST METRICS SERVICE
def test_metrics_service():
    # Test Risk levels
    assert get_volatility_risk_level(1.5) == "LOW"
    assert get_volatility_risk_level(2.5) == "MEDIUM"
    assert get_volatility_risk_level(4.0) == "HIGH"
    
    # Test formatting
    assert format_metric_value(0.1234, "percent") == "12.34%"
    assert format_metric_value(1.94, "ratio") == "1.94"
    assert format_metric_value(50, "count") == "50"

# 5. TEST PLOTLY CHARTS RENDER
def test_charts_render(mock_features, mock_signals):
    fig_cand = create_candlestick_chart(mock_features, mock_signals, "ASIANPAINT.NS", "dark")
    assert isinstance(fig_cand, go.Figure)
    
    equity_df = pd.DataFrame([{"date": "2026-01-01", "equity": 100000.0}])
    fig_equity = create_equity_curve_chart(equity_df, mock_features, "dark")
    assert isinstance(fig_equity, go.Figure)
    
    feat_imp = pd.DataFrame([{"feature": "rsi_14", "mean_importance": 0.10}])
    fig_imp = create_feature_importance_chart(feat_imp, "dark")
    assert isinstance(fig_imp, go.Figure)
    
    fig_spark = create_sparkline(pd.Series([100, 101, 102]), "dark")
    assert isinstance(fig_spark, go.Figure)
    
    fig_hist = create_probability_histograms(mock_signals, "dark")
    assert isinstance(fig_hist, go.Figure)
    
    fig_gauge = create_confidence_gauge(75.0, "BUY", "dark")
    assert isinstance(fig_gauge, go.Figure)
    
    fig_timeline = create_signal_history_timeline(mock_signals, "ASIANPAINT.NS", "2026-01-02", "dark")
    assert isinstance(fig_timeline, go.Figure)
    
    fig_momentum = create_momentum_charts(mock_features, "ASIANPAINT.NS", "2026-01-02", "dark")
    assert isinstance(fig_momentum, go.Figure)

# 6. TEST TRADE LOG FILTERS
def test_trade_log_reconstruction_and_filters(mock_trade_log):
    roundtrips = get_roundtrip_trades(mock_trade_log)
    assert len(roundtrips) == 1
    
    trade = roundtrips.iloc[0]
    assert trade["ticker"] == "ASIANPAINT.NS"
    assert trade["entry_date"] == "2026-01-01"
    assert trade["exit_date"] == "2026-01-02"
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 105.0
    assert trade["pnl"] == 50.0
    assert trade["return_pct"] == pytest.approx(0.05)
    assert trade["holding_days"] == 1

# 7. TEST NAVBAR CSS GENERATION
def test_navbar_css_generation():
    from src.dashboard.components import get_navbar_css
    css = get_navbar_css(3, "dark")
    assert isinstance(css, str)
    assert "<style>" in css
    assert "nth-child(3)" in css

# 8. TEST FONT STACK CONSTANT
def test_font_stack_constant():
    from src.dashboard.components import FONT_STACK
    assert isinstance(FONT_STACK, str)
    assert "JetBrains Mono" in FONT_STACK
