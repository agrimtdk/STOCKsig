import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def compute_ticker_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators for a single ticker's daily price dataframe.
    Assumes df is sorted chronologically.
    """
    df = df.copy()
    
    # Ensure standard column names and types
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']

    # --- Trend Indicators ---
    df['sma_5'] = close.rolling(window=5).mean()
    df['sma_10'] = close.rolling(window=10).mean()
    df['sma_20'] = close.rolling(window=20).mean()
    df['ema_10'] = close.ewm(span=10, adjust=False).mean()
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()

    # --- Momentum Indicators ---
    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use Wilder's EMA for smoothing
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    df['rsi_14'] = np.where((avg_gain == 0) & (avg_loss == 0), 50.0, rsi)

    # MACD
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # --- Volatility Indicators ---
    # Bollinger Bands (20-day SMA +/- 2 std)
    bb_mid = df['sma_20']
    bb_std = close.rolling(window=20).std()
    df['bollinger_upper'] = bb_mid + 2 * bb_std
    df['bollinger_lower'] = bb_mid - 2 * bb_std
    df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / (bb_mid + 1e-9)

    # ATR 14
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(window=14).mean()

    # Returns
    df['return_1d'] = close.pct_change(1)
    df['return_5d'] = close.pct_change(5)
    df['return_10d'] = close.pct_change(10)

    # Daily Volatility 20d (rolling std of 1d returns)
    df['daily_volatility_20d'] = df['return_1d'].rolling(window=20).std()

    # --- Volume Indicators ---
    # OBV (On-Balance Volume)
    direction = np.sign(delta)
    direction.iloc[0] = 0
    df['obv'] = (volume * direction).cumsum()

    # Volume Change
    df['volume_change'] = volume.pct_change(1)

    # Volume Z-Score 20d
    vol_mean = volume.rolling(window=20).mean()
    vol_std = volume.rolling(window=20).std()
    df['volume_zscore_20'] = (volume - vol_mean) / (vol_std + 1e-9)

    return df

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical indicators for the entire price dataframe.
    Sorts by ticker and date, groups by ticker to compute, and returns results.
    """
    if df.empty:
        logger.warning("Empty dataframe passed to compute_technical_indicators.")
        return df
        
    required_cols = {'ticker', 'date', 'open', 'high', 'low', 'close', 'volume'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for indicators: {missing}")

    # Ensure chronological order per ticker
    df_sorted = df.sort_values(by=['ticker', 'date']).copy()
    
    # Reset index to avoid pandas apply index matching bugs
    df_sorted = df_sorted.reset_index(drop=True)
    
    logger.info("Computing technical indicators grouped by ticker...")
    result_df = df_sorted.groupby('ticker', group_keys=False).apply(compute_ticker_technicals)
    
    return result_df
