import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.config import config
from src.features.technicals import compute_technical_indicators
from src.features.label_generator import generate_labels, compute_ticker_labels
from src.features.sentiment_features import (
    SentimentExtractor,
    extract_news_sentiments,
    extract_transcript_sentiments,
    get_macro_features
)

def test_technical_indicators():
    """Validates that technical indicators are calculated correctly grouped by ticker."""
    # Seed mock data for two tickers
    dates = pd.date_range(start="2025-01-01", end="2025-01-30", freq="D")
    n_days = len(dates)
    
    # Create simple trend (e.g. linear rise)
    close_prices = np.linspace(100.0, 130.0, n_days)
    high_prices = close_prices + 2.0
    low_prices = close_prices - 2.0
    volume = np.random.randint(1000, 2000, n_days)
    
    df1 = pd.DataFrame({
        "ticker": "TCS.NS",
        "date": dates.strftime('%Y-%m-%d'),
        "open": close_prices - 0.5,
        "high": high_prices,
        "low": low_prices,
        "close": close_prices,
        "volume": volume
    })
    
    # Second ticker has a flat trend
    close_prices_flat = np.ones(n_days) * 50.0
    df2 = pd.DataFrame({
        "ticker": "INFY.NS",
        "date": dates.strftime('%Y-%m-%d'),
        "open": close_prices_flat,
        "high": close_prices_flat + 1.0,
        "low": close_prices_flat - 1.0,
        "close": close_prices_flat,
        "volume": volume
    })
    
    df = pd.concat([df1, df2]).reset_index(drop=True)
    
    # Compute indicators
    res = compute_technical_indicators(df)
    
    # Verify indicator columns exist
    expected_cols = [
        'sma_5', 'sma_10', 'sma_20', 'ema_10', 'ema_20',
        'rsi_14', 'macd', 'macd_signal', 'macd_hist',
        'bollinger_upper', 'bollinger_lower', 'bollinger_width',
        'atr_14', 'daily_volatility_20d', 'obv', 'volume_change', 'volume_zscore_20'
    ]
    for col in expected_cols:
        assert col in res.columns
        
    # Verify that indicators are computed separately per ticker (e.g. SMA_5 of INFY.NS should be exactly 50)
    infy_rows = res[res['ticker'] == 'INFY.NS']
    # After row index 5, SMA_5 of flat series should be exactly 50
    assert np.isclose(infy_rows['sma_5'].iloc[10], 50.0)
    
    # Verify RSI behavior (TCS is rising, so RSI should be high, INFY is flat so RSI should be around 50)
    tcs_rows = res[res['ticker'] == 'TCS.NS']
    assert tcs_rows['rsi_14'].iloc[-1] > 70.0
    assert np.isclose(infy_rows['rsi_14'].iloc[-1], 50.0, atol=1.0)

def test_look_ahead_bias_leakage():
    """
    Spot checks for look-ahead bias leakage.
    Modifying a future close price must NOT change technical indicator values on earlier dates.
    """
    dates = pd.date_range(start="2025-01-01", end="2025-01-25", freq="D")
    n_days = len(dates)
    
    close_prices = np.linspace(100.0, 124.0, n_days)
    df = pd.DataFrame({
        "ticker": "TCS.NS",
        "date": dates.strftime('%Y-%m-%d'),
        "open": close_prices,
        "high": close_prices + 1.0,
        "low": close_prices - 1.0,
        "close": close_prices,
        "volume": 1000
    })
    
    # Base calculation
    res_base = compute_technical_indicators(df)
    
    # Create copy and change price at index 15 (representing day 16)
    df_modified = df.copy()
    df_modified.loc[15, 'close'] = 999.0
    df_modified.loc[15, 'high'] = 1000.0
    df_modified.loc[15, 'low'] = 998.0
    
    res_modified = compute_technical_indicators(df_modified)
    
    # Verify that for indices < 15, indicators are EXACTLY identical
    # If there is look-ahead bias, earlier indicators will shift.
    for col in ['sma_5', 'ema_10', 'rsi_14', 'macd', 'atr_14']:
        pd.testing.assert_series_equal(
            res_base.loc[:14, col], 
            res_modified.loc[:14, col], 
            obj=f"Look-ahead bias detected in {col}!"
        )

def test_label_generation_logic():
    """Validates label logic boundaries (BUY=1 for >+2%, SELL=-1 for <-2%, HOLD=0 otherwise)."""
    close = [100.0] * 10
    # Create simple data
    df = pd.DataFrame({
        "ticker": "TCS.NS",
        "date": [f"2025-01-{i:02d}" for i in range(1, 11)],
        "close": close
    })
    
    # 1. Test HOLD (flat series, 5-day return is 0.0%)
    res_hold = generate_labels(df)
    assert res_hold['label'].iloc[0] == 0
    assert pd.isna(res_hold['label'].iloc[-3]) # warm-down NaNs at the end
    
    # 2. Test BUY (+5% return)
    df_buy = df.copy()
    df_buy.loc[5, 'close'] = 105.0 # Day 6 (index 5) is 105 (+5% from Day 1 close of 100)
    res_buy = generate_labels(df_buy)
    assert res_buy['label'].iloc[0] == 1 # Day 1 close=100, Day 6 close=105 -> return = +5% -> BUY (1)
    
    # 3. Test SELL (-3% return)
    df_sell = df.copy()
    df_sell.loc[5, 'close'] = 97.0 # Day 6 is 97 (-3% from Day 1 close of 100)
    res_sell = generate_labels(df_sell)
    assert res_sell['label'].iloc[0] == -1 # Day 1 close=100, Day 6 close=97 -> return = -3% -> SELL (-1)

def test_news_sentiment_aggregation():
    """Validates daily news sentiment scoring and aggregation."""
    extractor = SentimentExtractor(use_finbert=False) # Force VADER fallback
    
    news_data = pd.DataFrame([
        {
            "id": "1",
            "ticker": "TCS.NS",
            "date": "2025-01-01",
            "headline": "TCS reports excellent and amazing profit increase",
            "summary": "Superb growth in digital assets boosts revenue to a record high.",
            "source": "Reuters",
            "url": "http://reuters.com/1"
        },
        {
            "id": "2",
            "ticker": "TCS.NS",
            "date": "2025-01-01",
            "headline": "TCS signs superb and great deal with UK retailer",
            "summary": "Fabulous expansion contract set to begin next month with success.",
            "source": "Bloomberg",
            "url": "http://bloomberg.com/2"
        },
        {
            "id": "3",
            "ticker": "INFY.NS",
            "date": "2025-01-01",
            "headline": "Infosys reports bad results and crisis",
            "summary": "Supply chain collapse is terrible and disastrous for revenue.",
            "source": "Reuters",
            "url": "http://reuters.com/3"
        }
    ])
    
    agg = extract_news_sentiments(news_data, extractor)
    
    # Verify aggregation structure
    assert len(agg) == 2 # TCS.NS and INFY.NS
    assert 'avg_news_sentiment' in agg.columns
    assert 'news_count' in agg.columns
    
    # TCS has 2 articles, INFY has 1
    tcs_row = agg[agg['ticker'] == 'TCS.NS'].iloc[0]
    infy_row = agg[agg['ticker'] == 'INFY.NS'].iloc[0]
    
    assert tcs_row['news_count'] == 2
    assert infy_row['news_count'] == 1
    
    # TCS should have positive sentiment (stellar profit, record deal)
    # INFY should have negative sentiment (decline, delays, impact)
    assert tcs_row['avg_news_sentiment'] > 0.1
    assert infy_row['avg_news_sentiment'] < -0.1

def test_transcript_stepwise_forward_fill():
    """Validates transcript stepwise mapping and forward filling boundary logic."""
    transcripts_df = pd.DataFrame([
        {
            "ticker": "TCS.NS",
            "date": "2025-01-10",
            "content": "CEO remarks: We have delivered strong growth and profit records this quarter. We expand our capacity."
        },
        {
            "ticker": "TCS.NS",
            "date": "2025-01-20",
            "content": "CFO statement: We report decline in operating margins and volatile costs. A weak performance."
        }
    ])
    
    # Calculate transcript features
    t_feat = extract_transcript_sentiments(transcripts_df)
    
    # Base daily frame to merge on
    dates = pd.date_range(start="2025-01-05", end="2025-01-25", freq="D")
    df = pd.DataFrame({
        "ticker": "TCS.NS",
        "date": dates.strftime('%Y-%m-%d')
    })
    
    # Merge and forward fill
    merged = pd.merge(df, t_feat, on=['ticker', 'date'], how='left')
    
    # Check that initially (days 5-9) they are NaN
    assert merged.loc[0:4, 'transcript_sentiment'].isnull().all()
    
    # Perform forward fill
    merged['transcript_sentiment'] = merged['transcript_sentiment'].ffill()
    
    # Check that after Jan 10 (day 6 onward, index 5), it is filled with first transcript sentiment (positive)
    assert merged.loc[5, 'transcript_sentiment'] > 0.0
    assert merged.loc[14, 'transcript_sentiment'] > 0.0 # Jan 19
    
    # Check that after Jan 20 (day 16 onward, index 15), it is filled with second transcript sentiment (negative)
    # Since VADER compound score for 'decline', 'volatile', 'weak' is negative
    assert merged.loc[15, 'transcript_sentiment'] < 0.0
