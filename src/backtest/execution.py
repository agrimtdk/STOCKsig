import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FEE_RATE = 0.0015  # 0.10% brokerage + 0.05% slippage per side (0.30% roundtrip)
POSITION_SIZE_PCT = 0.10  # 10% of current equity
MAX_EXPOSURE_CAP = 0.80  # 80% gross exposure limit

def execute_signal_day(
    date: str,
    prev_date: str,
    signals: pd.DataFrame,
    open_prices: dict,
    adj_close_prices: dict,
    portfolio,
):
    """
    Executes day t signals on day t+1 Open (which is 'date').
    - prev_date: day t (signal generation date)
    - date: day t+1 (execution date)
    - signals: DataFrame containing day t signals with ticker index
    - open_prices: dict mapping ticker -> day t+1 Open price
    - adj_close_prices: dict mapping ticker -> day t adj_close price (for valuation/exposure)
    - portfolio: BacktestPortfolio object
    """
    # 1. PROCESS EXITS (SELL signals)
    # Get tickers with active positions that have a SELL signal on prev_date
    exits = []
    for ticker, pos in list(portfolio.positions.items()):
        if ticker in signals.index:
            sig = signals.loc[ticker, "signal"]
            if sig == -1:  # SELL signal
                exits.append(ticker)

    for ticker in exits:
        if ticker not in open_prices or pd.isna(open_prices[ticker]):
            logger.warning(f"No open price for {ticker} on {date} to execute exit. Skipping.")
            continue
        
        raw_price = open_prices[ticker]
        exit_price = raw_price * (1 - FEE_RATE)
        pos = portfolio.positions[ticker]
        shares = pos["shares"]
        entry_price = pos["entry_price"]
        entry_date = pos["entry_date"]
        
        cash_received = shares * exit_price
        fee_paid = shares * raw_price * FEE_RATE
        realized_pnl = shares * (exit_price - entry_price)
        
        # Update portfolio cash and remove position
        portfolio.cash += cash_received
        portfolio.realized_pnl += realized_pnl
        del portfolio.positions[ticker]
        
        # Log trade
        trade_record = {
            "ticker": ticker,
            "action": "SELL",
            "date": date,
            "shares": shares,
            "price": exit_price,
            "raw_price": raw_price,
            "fee": fee_paid,
            "value": cash_received,
            "equity": portfolio.get_equity(adj_close_prices),  # equity before new entries
            "realized_pnl": realized_pnl,
            "holding_period_days": (pd.to_datetime(date) - pd.to_datetime(entry_date)).days,
        }
        portfolio.trade_log.append(trade_record)
        logger.info(f"Executed SELL for {ticker} on {date} at {exit_price:.2f} (raw Open: {raw_price:.2f}). PnL: {realized_pnl:.2f}")

    # 2. PROCESS ENTRIES (BUY signals)
    # Get BUY signals from prev_date
    buy_signals = signals[signals["signal"] == 1]
    if buy_signals.empty:
        return

    # Sort candidates by confidence_score in descending order (highest conviction first)
    buy_signals = buy_signals.sort_values(by="confidence_score", ascending=False)

    # Current equity from prev_date close (used for sizing and exposure check)
    current_equity = portfolio.get_equity(adj_close_prices)
    if current_equity <= 0:
        logger.warning(f"Negative or zero equity {current_equity:.2f} on {date}. Cannot execute BUYs.")
        return

    # Sizing capital per position (10% of previous close equity)
    target_position_value = current_equity * POSITION_SIZE_PCT

    for ticker, row in buy_signals.iterrows():
        # Check if we already have an active position
        if ticker in portfolio.positions:
            continue
            
        # Check if we have price data to execute
        if ticker not in open_prices or pd.isna(open_prices[ticker]):
            logger.warning(f"No open price for {ticker} on {date} to execute entry. Skipping.")
            continue

        raw_price = open_prices[ticker]
        entry_price = raw_price * (1 + FEE_RATE)

        # Enforce gross exposure cap <= 80%
        # Calculate current gross exposure value (using open prices of day t+1 for the check)
        # For existing positions, we value them at day t+1 Open as well.
        # Wait, for the exposure cap check, the constraint is: gross_exposure <= 0.80
        # Let's value current open positions at day t+1 open, and add new position value.
        current_exposure_val = 0.0
        for pos_ticker, pos_info in portfolio.positions.items():
            if pos_ticker in open_prices and not pd.isna(open_prices[pos_ticker]):
                current_exposure_val += pos_info["shares"] * open_prices[pos_ticker]
            elif pos_ticker in adj_close_prices and not pd.isna(adj_close_prices[pos_ticker]):
                # Fallback to prev close if open not available
                current_exposure_val += pos_info["shares"] * adj_close_prices[pos_ticker]

        max_allowed_val = current_equity * MAX_EXPOSURE_CAP
        capacity = max_allowed_val - current_exposure_val

        if capacity <= 0:
            logger.info(f"Gross exposure cap reached on {date}. Skipping BUY for {ticker}.")
            portfolio.skipped_signals_count += 1
            continue

        # Target size is 10% of equity, but capped by remaining exposure capacity and cash
        allocated_capital = target_position_value
        if allocated_capital > capacity:
            allocated_capital = capacity
            logger.info(f"Truncating trade size for {ticker} on {date} to {allocated_capital:.2f} to respect gross exposure cap.")

        # Cap by available cash
        allocated_capital = min(allocated_capital, portfolio.cash)
        if allocated_capital <= 0:
            logger.info(f"No cash available to execute BUY for {ticker} on {date}. Skipping.")
            portfolio.skipped_signals_count += 1
            continue

        # Calculate shares
        shares = int(np.floor(allocated_capital / entry_price))
        if shares <= 0:
            logger.info(f"Allocated capital too small for 1 share of {ticker} on {date}. Skipping.")
            portfolio.skipped_signals_count += 1
            continue

        cash_spent = shares * entry_price
        fee_paid = shares * raw_price * FEE_RATE

        # Update portfolio cash and positions
        portfolio.cash -= cash_spent
        portfolio.positions[ticker] = {
            "shares": shares,
            "entry_price": entry_price,
            "entry_date": date,
        }

        # Log trade
        trade_record = {
            "ticker": ticker,
            "action": "BUY",
            "date": date,
            "shares": shares,
            "price": entry_price,
            "raw_price": raw_price,
            "fee": fee_paid,
            "value": cash_spent,
            "equity": current_equity,
            "realized_pnl": 0.0,
            "holding_period_days": 0,
        }
        portfolio.trade_log.append(trade_record)
        logger.info(f"Executed BUY for {ticker} on {date} at {entry_price:.2f} (raw Open: {raw_price:.2f}). Shares: {shares}")
