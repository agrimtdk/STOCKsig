import pandas as pd
import numpy as np
import yfinance as yf
import json
import logging
import os

logger = logging.getLogger(__name__)

def fetch_nifty50(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Attempts to fetch NIFTY 50 (^NSEI) from yfinance for the specified date range.
    Returns a DataFrame with 'date' and 'close' columns, or empty DataFrame if failed.
    """
    try:
        logger.info(f"Attempting to download ^NSEI from yfinance from {start_date} to {end_date}...")
        # Add a buffer day to ensure coverage
        start_dt = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        end_dt = (pd.to_datetime(end_date) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        
        ticker = yf.Ticker("^NSEI")
        df = ticker.history(start=start_dt, end=end_dt)
        if df.empty:
            logger.warning("yfinance returned empty DataFrame for ^NSEI.")
            return pd.DataFrame()
            
        df = df.reset_index()
        # Handle Date timezone
        if "Date" in df.columns:
            df["date"] = df["Date"].dt.strftime("%Y-%m-%d")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
        df = df.rename(columns={"Close": "close"})
        df = df[["date", "close"]].dropna()
        
        # Filter to our exact period
        df = df[(df["date"] >= start_date) & (df["date"] <= end_date)].sort_values("date")
        return df
    except Exception as e:
        logger.error(f"Error fetching ^NSEI from yfinance: {e}")
        return pd.DataFrame()

def build_fallback_benchmark(features_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Builds an equal-weighted buy-and-hold index of the 10 universe stocks as a fallback.
    Uses adj_close from features.parquet.
    """
    logger.info("Building fallback equal-weighted buy-and-hold benchmark from universe stocks...")
    try:
        df_feat = pd.read_parquet(features_path)
        # Reset index if ticker and date are indices
        if "ticker" in df_feat.index.names and "date" in df_feat.index.names:
            df_feat = df_feat.reset_index()
            
        df_feat["date"] = df_feat["date"].astype(str)
        # Filter dates
        df_feat = df_feat[(df_feat["date"] >= start_date) & (df_feat["date"] <= end_date)]
        
        # Pivot to get adj_close for each ticker
        df_pivot = df_feat.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        
        # Drop any row that doesn't have all tickers if possible, or forward fill
        df_pivot = df_pivot.ffill().bfill()
        
        if df_pivot.empty:
            logger.error("Pivot of features data for fallback benchmark is empty.")
            return pd.DataFrame()
            
        # Equal-weighted portfolio value starting at 100,000 on start_date
        # Allocating 10,000 to each of the 10 stocks on first day
        first_row = df_pivot.iloc[0]
        shares = 10000.0 / first_row
        
        # Daily value
        df_benchmark = pd.DataFrame(index=df_pivot.index)
        df_benchmark["close"] = df_pivot.multiply(shares).sum(axis=1)
        df_benchmark = df_benchmark.reset_index()
        
        logger.info(f"Fallback benchmark built successfully. Tickers: {list(df_pivot.columns)}")
        return df_benchmark
    except Exception as e:
        logger.error(f"Error building fallback benchmark: {e}")
        return pd.DataFrame()

def calculate_benchmark_metrics(benchmark_df: pd.DataFrame, initial_capital: float = 100000.0) -> dict:
    """
    Calculates returns, CAGR, Sharpe, and Max Drawdown for the benchmark.
    """
    if benchmark_df.empty:
        return {"total_return": 0.0, "cagr": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    # Scale benchmark close to start at initial_capital
    first_val = benchmark_df["close"].iloc[0]
    scaled_equity = benchmark_df["close"] * (initial_capital / first_val)
    
    total_return = (scaled_equity.iloc[-1] - initial_capital) / initial_capital
    
    start_date = pd.to_datetime(benchmark_df["date"].iloc[0])
    end_date = pd.to_datetime(benchmark_df["date"].iloc[-1])
    days = (end_date - start_date).days
    years = max(days / 365.25, 1.0 / 365.25)
    cagr = (scaled_equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0
    
    daily_returns = scaled_equity.pct_change().dropna()
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
    else:
        sharpe = 0.0
        
    roll_max = scaled_equity.cummax()
    drawdowns = (scaled_equity - roll_max) / roll_max
    max_dd = float(drawdowns.min())
    
    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
    }

def run_benchmark_comparison(
    strategy_metrics: dict,
    equity_curve_df: pd.DataFrame,
    features_path: str,
    output_path: str
) -> dict:
    """
    Orchestrates the benchmarking process and saves the comparison JSON.
    """
    start_date = equity_curve_df["date"].iloc[0]
    end_date = equity_curve_df["date"].iloc[-1]
    
    # 1. Attempt yfinance download of ^NSEI
    benchmark_df = fetch_nifty50(start_date, end_date)
    benchmark_name = "^NSEI (NIFTY 50)"
    
    # 2. Fallback to equal-weighted basket if yfinance fails or is offline
    if benchmark_df.empty:
        benchmark_df = build_fallback_benchmark(features_path, start_date, end_date)
        benchmark_name = "Equal-Weighted Basket (10 Stocks Fallback)"
        
    if benchmark_df.empty:
        logger.error("Both NIFTY 50 and fallback benchmark failed. Cannot perform comparison.")
        return {}

    # Calculate benchmark metrics
    bench_metrics = calculate_benchmark_metrics(benchmark_df, initial_capital=100000.0)
    
    # Construct comparison dictionary
    comparison = {
        "benchmark_name": benchmark_name,
        "start_date": start_date,
        "end_date": end_date,
        "strategy": {
            "total_return": strategy_metrics["total_return"],
            "cagr": strategy_metrics["cagr"],
            "sharpe_ratio": strategy_metrics["sharpe_ratio"],
            "max_drawdown": strategy_metrics["max_drawdown"],
        },
        "benchmark": bench_metrics,
        "comparison": {
            "alpha_return": strategy_metrics["total_return"] - bench_metrics["total_return"],
            "alpha_cagr": strategy_metrics["cagr"] - bench_metrics["cagr"],
        }
    }
    
    # Ensure folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=4)
        
    logger.info(f"Saved benchmark comparison to {output_path}")
    return comparison
