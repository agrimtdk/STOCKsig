import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_all_metrics(
    equity_curve_df: pd.DataFrame,
    trade_log_df: pd.DataFrame,
    total_buy_signals: int
) -> dict:
    """
    Computes all returns, risk, trade, and diagnostic metrics for the backtest.
    """
    metrics = {}

    # --- RETURN METRICS ---
    starting_equity = float(equity_curve_df["equity"].iloc[0])
    ending_equity = float(equity_curve_df["equity"].iloc[-1])
    total_return = (ending_equity - starting_equity) / starting_equity

    # CAGR Calculation
    start_date = pd.to_datetime(equity_curve_df["date"].iloc[0])
    end_date = pd.to_datetime(equity_curve_df["date"].iloc[-1])
    days = (end_date - start_date).days
    years = max(days / 365.25, 1.0 / 365.25)
    cagr = (ending_equity / starting_equity) ** (1.0 / years) - 1.0

    metrics["total_return"] = total_return
    metrics["cagr"] = cagr

    # --- RISK METRICS ---
    daily_returns = equity_curve_df["equity"].pct_change().dropna()
    
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        annualized_volatility = float(daily_returns.std() * np.sqrt(252))
        sharpe_ratio = float((daily_returns.mean() / daily_returns.std()) * np.sqrt(252))
    else:
        annualized_volatility = 0.0
        sharpe_ratio = 0.0

    # Sortino Ratio
    downside_returns = daily_returns[daily_returns < 0]
    if len(daily_returns) > 0 and len(downside_returns) > 0:
        downside_std = downside_returns.std()
        if downside_std > 0:
            sortino_ratio = float((daily_returns.mean() / downside_std) * np.sqrt(252))
        else:
            sortino_ratio = 0.0
    else:
        sortino_ratio = 0.0

    # Max Drawdown
    roll_max = equity_curve_df["equity"].cummax()
    drawdowns = (equity_curve_df["equity"] - roll_max) / roll_max
    max_drawdown = float(drawdowns.min())

    metrics["volatility"] = annualized_volatility
    metrics["sharpe_ratio"] = sharpe_ratio
    metrics["sortino_ratio"] = sortino_ratio
    metrics["max_drawdown"] = max_drawdown

    # --- TRADE METRICS ---
    # Filter for exits to calculate closed trade metrics
    if not trade_log_df.empty:
        exits = trade_log_df[trade_log_df["action"].str.startswith(("SELL", "LIQUIDATE"))].copy()
    else:
        exits = pd.DataFrame()

    total_trades = len(exits)
    metrics["total_trades"] = total_trades

    if total_trades > 0:
        wins = exits[exits["realized_pnl"] > 0]
        losses = exits[exits["realized_pnl"] < 0]
        
        win_rate = len(wins) / total_trades
        avg_win = float(wins["realized_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses["realized_pnl"].mean()) if len(losses) > 0 else 0.0
        
        gross_profits = float(wins["realized_pnl"].sum())
        gross_losses = float(abs(losses["realized_pnl"].sum()))
        
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else (999.0 if gross_profits > 0 else 0.0)
        expectancy = float(exits["realized_pnl"].mean())
        
        avg_holding_period = float(exits["holding_period_days"].mean())
    else:
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        expectancy = 0.0
        avg_holding_period = 0.0

    metrics["win_rate"] = win_rate
    metrics["avg_win"] = avg_win
    metrics["avg_loss"] = avg_loss
    metrics["profit_factor"] = profit_factor
    metrics["expectancy"] = expectancy

    # --- DIAGNOSTIC METRICS ---
    metrics["average_holding_period"] = avg_holding_period

    # Signal-to-trade conversion
    if not trade_log_df.empty:
        total_buys = len(trade_log_df[trade_log_df["action"] == "BUY"])
    else:
        total_buys = 0

    conversion_rate = total_buys / total_buy_signals if total_buy_signals > 0 else 0.0
    metrics["signal_to_trade_conversion_rate"] = conversion_rate

    # Exposure utilization
    mean_exposure = float(equity_curve_df["gross_exposure"].mean())
    metrics["mean_exposure_utilization"] = mean_exposure

    logger.info(f"Metrics computation complete. Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Drawdown: {max_drawdown:.2%}")
    return metrics
