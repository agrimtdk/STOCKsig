import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

FEE_RATE = 0.0015  # 0.15% fee per side

class BacktestPortfolio:
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # ticker -> {"shares": int, "entry_price": float, "entry_date": str}
        self.realized_pnl = 0.0
        self.equity_curve = []  # List of daily metrics dicts
        self.trade_log = []  # List of executed trade dicts
        
        # Diagnostics
        self.skipped_signals_count = 0
        self.total_signals_count = 0
        
        # Keep track of last known prices for valuation fallback
        self.last_known_prices = {}

    def get_equity(self, adj_close_prices: dict) -> float:
        """
        Calculates total equity based on current cash and market value of positions.
        """
        positions_value = 0.0
        for ticker, pos in self.positions.items():
            price = adj_close_prices.get(ticker, self.last_known_prices.get(ticker, pos["entry_price"]))
            if not pd.isna(price):
                self.last_known_prices[ticker] = price
                positions_value += pos["shares"] * price
            else:
                positions_value += pos["shares"] * pos["entry_price"]
        return self.cash + positions_value

    def update_daily_state(self, date: str, adj_close_prices: dict):
        """
        Computes and records daily valuation metrics at Close.
        Uses adj_close for valuation.
        """
        positions_value = 0.0
        unrealized_pnl = 0.0
        
        for ticker, pos in self.positions.items():
            price = adj_close_prices.get(ticker, self.last_known_prices.get(ticker, pos["entry_price"]))
            if not pd.isna(price):
                self.last_known_prices[ticker] = price
                pos_val = pos["shares"] * price
                positions_value += pos_val
                # Unrealized PnL based on current valuation vs execution entry price
                unrealized_pnl += pos["shares"] * (price - pos["entry_price"])
            else:
                positions_value += pos["shares"] * pos["entry_price"]

        equity = self.cash + positions_value
        gross_exposure = positions_value / equity if equity > 0.0 else 0.0

        daily_record = {
            "date": date,
            "cash": self.cash,
            "positions_value": positions_value,
            "equity": equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "gross_exposure": gross_exposure,
        }
        self.equity_curve.append(daily_record)
        logger.debug(f"Daily valuation on {date}: Cash={self.cash:.2f}, PosVal={positions_value:.2f}, Equity={equity:.2f}, Exposure={gross_exposure*100:.2f}%")

    def force_liquidate_all(self, date: str, open_prices: dict, adj_close_prices: dict):
        """
        Forces liquidation of all open positions at the final backtest date.
        Uses final day's Open price, falling back to adj_close.
        """
        if not self.positions:
            return

        tickers_to_liquidate = list(self.positions.keys())
        logger.info(f"Force liquidating all open positions ({len(tickers_to_liquidate)}) on final date {date}...")

        for ticker in tickers_to_liquidate:
            pos = self.positions[ticker]
            shares = pos["shares"]
            entry_price = pos["entry_price"]
            entry_date = pos["entry_date"]

            # Select price: final available open, fallback to adj_close, fallback to entry_price
            price_source = "open"
            raw_price = open_prices.get(ticker, np.nan)
            if pd.isna(raw_price):
                raw_price = adj_close_prices.get(ticker, np.nan)
                price_source = "adj_close"
            if pd.isna(raw_price):
                raw_price = self.last_known_prices.get(ticker, entry_price)
                price_source = "last_known"

            exit_price = raw_price * (1 - FEE_RATE)
            cash_received = shares * exit_price
            fee_paid = shares * raw_price * FEE_RATE
            realized_pnl = shares * (exit_price - entry_price)

            self.cash += cash_received
            self.realized_pnl += realized_pnl
            del self.positions[ticker]

            # Log trade
            trade_record = {
                "ticker": ticker,
                "action": f"LIQUIDATE_{price_source.upper()}",
                "date": date,
                "shares": shares,
                "price": exit_price,
                "raw_price": raw_price,
                "fee": fee_paid,
                "value": cash_received,
                "equity": self.cash,  # post-liquidation equity
                "realized_pnl": realized_pnl,
                "holding_period_days": (pd.to_datetime(date) - pd.to_datetime(entry_date)).days,
            }
            self.trade_log.append(trade_record)
            logger.info(f"Liquidated {ticker} on final date {date} using {price_source} price {raw_price:.2f}. Exit: {exit_price:.2f}. PnL: {realized_pnl:.2f}")

    def save_equity_curve(self, filepath: str):
        """
        Saves the recorded daily equity curve to a Parquet file.
        """
        df = pd.DataFrame(self.equity_curve)
        # Ensure date column exists and is string format
        df["date"] = df["date"].astype(str)
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved equity curve Parquet to {filepath}")

    def save_trade_log(self, filepath: str):
        """
        Saves the executed trades log to a CSV file.
        """
        if self.trade_log:
            df = pd.DataFrame(self.trade_log)
        else:
            df = pd.DataFrame(columns=[
                "ticker", "action", "date", "shares", "price", "raw_price", 
                "fee", "value", "equity", "realized_pnl", "holding_period_days"
            ])
        df.to_csv(filepath, index=False)
        logger.info(f"Saved trade log CSV to {filepath}")
