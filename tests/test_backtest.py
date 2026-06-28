import pytest
import pandas as pd
import numpy as np
import os
import shutil

from src.backtest.backtester import run_backtest_pipeline
from src.backtest.portfolio import BacktestPortfolio
from src.backtest.execution import execute_signal_day, FEE_RATE

@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture to provide a clean temporary folder for test files."""
    test_dir = tmp_path / "backtest_test"
    test_dir.mkdir()
    yield test_dir

def create_mock_data(test_dir):
    """Creates small mock signals and features parquets for testing."""
    # 5 dates: Day 1 to Day 5
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    tickers = ["TICKER_A", "TICKER_B"]
    
    # Grid of ticker x date
    idx = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    
    # 1. Mock Features: open is 100 on all days, adj_close is 100
    df_feat = pd.DataFrame(index=idx)
    df_feat["open"] = 100.0
    df_feat["adj_close"] = 100.0
    df_feat["close"] = 100.0
    df_feat["high"] = 100.0
    df_feat["low"] = 100.0
    df_feat["volume"] = 10000.0
    
    # 2. Mock Signals:
    # TICKER_A: BUY on Day 1 (execute Day 2 Open), HOLD on others.
    # TICKER_B: BUY on Day 1 (execute Day 2 Open), SELL on Day 3 (execute Day 4 Open).
    df_sig = pd.DataFrame(index=idx)
    df_sig["signal"] = 0
    df_sig["confidence_score"] = 50.0
    df_sig["prob_sell"] = 0.20
    df_sig["prob_hold"] = 0.60
    df_sig["prob_buy"] = 0.20
    
    # Day 1 BUY signals
    df_sig.loc[("TICKER_A", "2026-01-01"), "signal"] = 1
    df_sig.loc[("TICKER_A", "2026-01-01"), "confidence_score"] = 90.0  # higher priority
    
    df_sig.loc[("TICKER_B", "2026-01-01"), "signal"] = 1
    df_sig.loc[("TICKER_B", "2026-01-01"), "confidence_score"] = 80.0
    
    # Day 3 SELL signal for B
    df_sig.loc[("TICKER_B", "2026-01-03"), "signal"] = -1
    
    # Save to parquet files
    feat_path = test_dir / "features.parquet"
    sig_path = test_dir / "signals.parquet"
    
    df_feat.to_parquet(feat_path)
    df_sig.to_parquet(sig_path)
    
    return str(sig_path), str(feat_path)

def test_backtest_pipeline_execution(temp_test_dir):
    """
    Validates next-day execution, no same-day fill, transaction costs,
    final-day liquidation, and output report generation.
    """
    sig_path, feat_path = create_mock_data(temp_test_dir)
    reports_dir = temp_test_dir / "reports"
    
    # Run backtest
    report = run_backtest_pipeline(
        signals_path=sig_path,
        features_path=feat_path,
        reports_dir=str(reports_dir),
        initial_capital=100000.0
    )
    
    # 1. Check file generation
    trade_log_file = reports_dir / "trade_log.csv"
    equity_curve_file = reports_dir / "equity_curve.parquet"
    report_file = reports_dir / "backtest_report.json"
    comparison_file = reports_dir / "benchmark_comparison.json"
    
    assert trade_log_file.exists()
    assert equity_curve_file.exists()
    assert report_file.exists()
    assert comparison_file.exists()
    
    # Load trade log and equity curve
    df_trades = pd.read_csv(trade_log_file)
    df_equity = pd.read_parquet(equity_curve_file)
    
    # 2. Verify Next-Day Execution & No Same-Day Fill
    # Signals generated on Day 1 (2026-01-01) must execute on Day 2 (2026-01-02) Open
    buy_trades = df_trades[df_trades["action"] == "BUY"]
    assert len(buy_trades) == 2
    assert all(buy_trades["date"] == "2026-01-02")
    
    # 3. Verify Transaction Costs Applied
    # Raw open price is 100.0, fee rate is 0.0015
    # Expected entry price = 100 * 1.0015 = 100.15
    for _, trade in buy_trades.iterrows():
        assert trade["raw_price"] == 100.0
        assert trade["price"] == pytest.approx(100.15)
        assert trade["fee"] == pytest.approx(trade["shares"] * 100.0 * FEE_RATE)
        
    # SELL signal on Day 3 (2026-01-03) must execute on Day 4 (2026-01-04) Open
    sell_trades = df_trades[df_trades["action"] == "SELL"]
    assert len(sell_trades) == 1
    assert sell_trades.iloc[0]["date"] == "2026-01-04"
    assert sell_trades.iloc[0]["ticker"] == "TICKER_B"
    assert sell_trades.iloc[0]["price"] == pytest.approx(99.85)  # 100 * 0.9985
    
    # 4. Verify Final Day Force Liquidation
    # Final date is 2026-01-05. TICKER_A is still open and must be liquidated.
    liq_trades = df_trades[df_trades["action"].str.startswith("LIQUIDATE")]
    assert len(liq_trades) == 1
    assert liq_trades.iloc[0]["ticker"] == "TICKER_A"
    assert liq_trades.iloc[0]["date"] == "2026-01-05"
    assert liq_trades.iloc[0]["price"] == pytest.approx(99.85)  # 100 * 0.9985
    
    # Verify ending positions value at final close is 0
    final_equity_row = df_equity.iloc[-1]
    assert final_equity_row["date"] == "2026-01-05"
    assert final_equity_row["positions_value"] == 0.0
    assert final_equity_row["equity"] == final_equity_row["cash"]

def test_exposure_constraints_and_non_negative_cash(temp_test_dir):
    """
    Validates that:
    1. Cash is never negative.
    2. Gross exposure <= 80% is strictly enforced during entry sizing.
    """
    # Create mock parquets where positions are extremely large or we try to buy too many
    # 5 dates
    dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
    # 11 tickers (more than 80% capacity if we buy all with 10% sizing)
    tickers = [f"TICKER_{i}" for i in range(1, 12)]
    
    idx = pd.MultiIndex.from_product([tickers, dates], names=["ticker", "date"])
    
    df_feat = pd.DataFrame(index=idx)
    df_feat["open"] = 100.0
    df_feat["adj_close"] = 100.0
    df_feat["close"] = 100.0
    df_feat["high"] = 100.0
    df_feat["low"] = 100.0
    df_feat["volume"] = 10000.0
    
    df_sig = pd.DataFrame(index=idx)
    df_sig["signal"] = 0
    df_sig["confidence_score"] = 50.0
    df_sig["prob_sell"] = 0.20
    df_sig["prob_hold"] = 0.60
    df_sig["prob_buy"] = 0.20
    
    # Generate BUY signals for ALL 11 tickers on Day 1
    for i, ticker in enumerate(tickers):
        df_sig.loc[(ticker, "2026-01-01"), "signal"] = 1
        # different confidence scores to sort them
        df_sig.loc[(ticker, "2026-01-01"), "confidence_score"] = 100.0 - i
        
    feat_path = temp_test_dir / "features_limit.parquet"
    sig_path = temp_test_dir / "signals_limit.parquet"
    df_feat.to_parquet(feat_path)
    df_sig.to_parquet(sig_path)
    
    reports_dir = temp_test_dir / "reports_limit"
    
    # Run backtest with 100,000 INR
    run_backtest_pipeline(
        signals_path=str(sig_path),
        features_path=str(feat_path),
        reports_dir=str(reports_dir),
        initial_capital=100000.0
    )
    
    df_trades = pd.read_csv(reports_dir / "trade_log.csv")
    df_equity = pd.read_parquet(reports_dir / "equity_curve.parquet")
    
    # Verify cash never negative
    assert (df_equity["cash"] >= 0).all()
    
    # Verify exposure never violates cap.
    # On day 2 (when buy orders are filled), total open positions should not exceed 9 (8 full + 1 truncated)
    # The 10th and 11th positions must be skipped entirely as capacity is fully exhausted.
    buy_trades = df_trades[df_trades["action"] == "BUY"]
    
    # Verify that gross exposure at close is <= 80%
    assert (df_equity["gross_exposure"] <= 0.80001).all()
    
    # Check that at most 9 positions were bought (meaning the rest were skipped)
    assert len(buy_trades) <= 9
    assert len(buy_trades) < len(tickers)
