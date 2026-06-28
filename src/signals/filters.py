import pandas as pd
import logging

logger = logging.getLogger(__name__)

def check_buy_confirmations(df: pd.DataFrame) -> pd.Series:
    """
    Checks technical indicators for BUY confirmations.
    Rule: close > sma_20 * 0.98 AND rsi_14 < 75 AND macd_hist > -0.05 AND volume_zscore_20 > -1.0
    Returns a boolean Series (True if confirmed).
    """
    # Enforce checks safely, handling potential missing columns
    required = {'close', 'sma_20', 'rsi_14', 'macd_hist', 'volume_zscore_20'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required indicator columns for BUY filter: {missing}")

    confirm_close = df['close'] > df['sma_20'] * 0.98
    confirm_rsi = df['rsi_14'] < 75
    confirm_macd = df['macd_hist'] > -0.05
    confirm_vol = df['volume_zscore_20'] > -1.0
    
    return confirm_close & confirm_rsi & confirm_macd & confirm_vol

def check_sell_confirmations(df: pd.DataFrame) -> pd.Series:
    """
    Checks technical indicators for SELL confirmations.
    Rule: close < sma_20 * 1.02 AND rsi_14 > 25 AND macd_hist < 0.05
    Returns a boolean Series (True if confirmed).
    """
    required = {'close', 'sma_20', 'rsi_14', 'macd_hist'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required indicator columns for SELL filter: {missing}")

    confirm_close = df['close'] < df['sma_20'] * 1.02
    confirm_rsi = df['rsi_14'] > 25
    confirm_macd = df['macd_hist'] < 0.05
    
    return confirm_close & confirm_rsi & confirm_macd
