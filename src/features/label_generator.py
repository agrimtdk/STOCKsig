import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_ticker_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes labels for a single ticker's price dataframe.
    Assumes df is sorted chronologically.
    """
    df = df.copy()
    
    # 5 trading days look-ahead return
    # close.shift(-5) gets the close price 5 days into the future
    close_future = df['close'].shift(-5)
    close_current = df['close']
    
    df['future_return_5d'] = (close_future - close_current) / close_current
    
    # Label mapping:
    # BUY = 1 if future return > 2% (+0.02)
    # SELL = -1 if future return < -2% (-0.02)
    # HOLD = 0 otherwise
    # Keep NaNs as NaN (warm-down period at the end of data range)
    conditions = [
        df['future_return_5d'] > 0.02,
        df['future_return_5d'] < -0.02,
        df['future_return_5d'].notna()
    ]
    choices = [1, -1, 0]
    
    df['label'] = np.select(conditions, choices, default=np.nan)
    
    return df

def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates 5-day look-ahead returns and target labels.
    Sorts by ticker and date, groups by ticker to compute, and returns results.
    """
    if df.empty:
        logger.warning("Empty dataframe passed to generate_labels.")
        return df

    required_cols = {'ticker', 'date', 'close'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for labels: {missing}")

    df_sorted = df.sort_values(by=['ticker', 'date']).copy()
    df_sorted = df_sorted.reset_index(drop=True)

    logger.info("Generating target labels (BUY/HOLD/SELL) with 5-day prediction horizon...")
    result_df = df_sorted.groupby('ticker', group_keys=False).apply(compute_ticker_labels)
    
    return result_df
