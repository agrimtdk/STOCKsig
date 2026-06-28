import pandas as pd
import numpy as np
import logging
import os
import json

from src.backtest.portfolio import BacktestPortfolio
from src.backtest.execution import execute_signal_day
from src.backtest.metrics import compute_all_metrics
from src.backtest.benchmark import run_benchmark_comparison

logger = logging.getLogger(__name__)

def run_backtest_pipeline(
    signals_path: str,
    features_path: str,
    reports_dir: str,
    initial_capital: float = 100000.0
) -> dict:
    """
    Orchestrates the entire Phase 5 backtest pipeline.
    """
    logger.info("Initializing Backtest Orchestrator...")
    
    # 1. Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # 2. Load data
    df_sig = pd.read_parquet(signals_path)
    df_feat = pd.read_parquet(features_path)
    
    # Ensure correct index names
    if df_sig.index.names != ["ticker", "date"]:
        df_sig = df_sig.reset_index().set_index(["ticker", "date"])
    if df_feat.index.names != ["ticker", "date"]:
        df_feat = df_feat.reset_index().set_index(["ticker", "date"])
        
    # Standardize date indices to string format
    df_sig.index = df_sig.index.set_levels(df_sig.index.levels[1].astype(str), level="date")
    df_feat.index = df_feat.index.set_levels(df_feat.index.levels[1].astype(str), level="date")
    
    # Merge signals and features (retaining prices and confidence scores)
    df_merged = df_sig.join(df_feat[["open", "adj_close"]], how="inner").sort_index()
    logger.info(f"Merged signals and features. Total rows: {df_merged.shape[0]}")
    
    # Get sorted chronological unique dates
    dates = sorted(df_merged.index.get_level_values("date").unique())
    logger.info(f"Backtesting over {len(dates)} trading days: {dates[0]} to {dates[-1]}")
    
    # Initialize Portfolio
    portfolio = BacktestPortfolio(initial_capital=initial_capital)
    
    # Count total treatable buy signals (excluding the final date)
    treatable_df = df_merged[df_merged.index.get_level_values("date") != dates[-1]]
    total_buy_signals = int((treatable_df["signal"] == 1).sum())
    portfolio.total_signals_count = total_buy_signals
    logger.info(f"Total treatable BUY signals in dataset: {total_buy_signals}")
    
    # 3. Daily Chronological Simulation Loop
    for k, date in enumerate(dates):
        prev_date = dates[k-1] if k > 0 else None
        is_final_date = (k == len(dates) - 1)
        
        # Open prices on date for execution fills
        open_prices = df_merged.xs(date, level="date")["open"].to_dict()
        # Close prices on date for valuation
        adj_close_prices = df_merged.xs(date, level="date")["adj_close"].to_dict()
        
        # Execute signals from prev_date on date Open
        if prev_date is not None:
            signals_prev = df_merged.xs(prev_date, level="date")[["signal", "confidence_score"]].copy()
            
            # If final day, do not enter new BUY positions (only execute exits)
            if is_final_date:
                signals_prev = signals_prev[signals_prev["signal"] == -1]
                
            # Prices on prev_date close (used for sizing and portfolio valuation check)
            adj_close_prev = df_merged.xs(prev_date, level="date")["adj_close"].to_dict()
            
            execute_signal_day(
                date=date,
                prev_date=prev_date,
                signals=signals_prev,
                open_prices=open_prices,
                adj_close_prices=adj_close_prev,
                portfolio=portfolio
            )
            
        # Force liquidation of all positions on final date
        if is_final_date:
            portfolio.force_liquidate_all(date, open_prices, adj_close_prices)
            
        # Value portfolio at Close of date
        portfolio.update_daily_state(date, adj_close_prices)
        
    logger.info("Daily simulation loop completed.")
    
    # 4. Save results to files
    equity_curve_path = os.path.join(reports_dir, "equity_curve.parquet")
    trade_log_path = os.path.join(reports_dir, "trade_log.csv")
    portfolio.save_equity_curve(equity_curve_path)
    portfolio.save_trade_log(trade_log_path)
    
    # 5. Compute performance metrics
    equity_df = pd.DataFrame(portfolio.equity_curve)
    trade_df = pd.DataFrame(portfolio.trade_log)
    
    strategy_metrics = compute_all_metrics(equity_df, trade_df, total_buy_signals)
    
    # 6. Run benchmark comparison
    benchmark_path = os.path.join(reports_dir, "benchmark_comparison.json")
    benchmark_results = run_benchmark_comparison(
        strategy_metrics=strategy_metrics,
        equity_curve_df=equity_df,
        features_path=features_path,
        output_path=benchmark_path
    )
    
    # 7. Save final backtest report JSON
    report_path = os.path.join(reports_dir, "backtest_report.json")
    report = {
        "strategy_metrics": strategy_metrics,
        "benchmark_comparison": benchmark_results
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Saved final backtest report to {report_path}")
    return report
