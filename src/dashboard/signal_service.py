import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_latest_signal_info(signals_df: pd.DataFrame, ticker: str, date: str = None) -> dict:
    """
    Retrieves signal, confidence score, and class probabilities for a stock on a specific date.
    If date is None, uses the latest available date for that ticker.
    """
    if signals_df.empty or ticker not in signals_df.index.get_level_values("ticker"):
        return {}
        
    df_stock = signals_df.xs(ticker, level="ticker")
    if df_stock.empty:
        return {}
        
    if date is None or date not in df_stock.index:
        # Use latest date
        date = df_stock.index.max()
        
    row = df_stock.loc[date]
    
    # Map raw signal (-1, 0, 1) to label string
    sig_val = int(row["signal"])
    if sig_val == 1:
        label = "BUY"
    elif sig_val == -1:
        label = "SELL"
    else:
        label = "HOLD"
        
    return {
        "date": date,
        "ticker": ticker,
        "signal": sig_val,
        "label": label,
        "confidence_score": float(row["confidence_score"]),
        "prob_sell": float(row.get("prob_sell", 0.0)),
        "prob_hold": float(row.get("prob_hold", 0.0)),
        "prob_buy": float(row.get("prob_buy", 0.0)),
    }

def get_technical_snapshot(features_df: pd.DataFrame, ticker: str, date: str) -> dict:
    """
    Retrieves the technical indicator snapshot for a stock on a specific date.
    """
    if features_df.empty or ticker not in features_df.index.get_level_values("ticker"):
        return {}
        
    df_stock = features_df.xs(ticker, level="ticker")
    if date not in df_stock.index:
        return {}
        
    row = df_stock.loc[date]
    return {
        "rsi": float(row.get("rsi_14", 50.0)),
        "macd": float(row.get("macd", 0.0)),
        "macd_signal": float(row.get("macd_signal", 0.0)),
        "macd_hist": float(row.get("macd_hist", 0.0)),
        "atr": float(row.get("atr_14", 0.0)),
        "bollinger_width": float(row.get("bollinger_width", 0.0)),
        "volatility": float(row.get("daily_volatility_20d", 0.0)) * 100.0, # in percent
    }

def get_sentiment_snapshot(features_df: pd.DataFrame, ticker: str, date: str) -> dict:
    """
    Retrieves the news and transcript sentiment snapshot for a stock on a specific date.
    """
    if features_df.empty or ticker not in features_df.index.get_level_values("ticker"):
        return {}
        
    df_stock = features_df.xs(ticker, level="ticker")
    if date not in df_stock.index:
        return {}
        
    row = df_stock.loc[date]
    return {
        "news_sentiment": float(row.get("avg_news_sentiment", 0.0)),
        "news_count": int(row.get("news_count", 0)),
        "transcript_sentiment": float(row.get("transcript_sentiment", 0.0)),
    }

def get_market_regime(features_df: pd.DataFrame, ticker: str, date: str) -> dict:
    """
    Computes Trend, Volatility, and Sentiment regimes for a stock on a specific date.
    """
    if features_df.empty or ticker not in features_df.index.get_level_values("ticker"):
        return {"trend": "Neutral", "volatility": "Medium", "sentiment": "Neutral"}
        
    df_stock = features_df.xs(ticker, level="ticker")
    if date not in df_stock.index:
        return {"trend": "Neutral", "volatility": "Medium", "sentiment": "Neutral"}
        
    row = df_stock.loc[date]
    
    # 1. Trend Regime
    close = row.get("close", 100.0)
    sma_20 = row.get("sma_20", 100.0)
    macd_hist = row.get("macd_hist", 0.0)
    
    if close > sma_20 * 1.005 and macd_hist > 0:
        trend = "Bullish"
    elif close < sma_20 * 0.995 and macd_hist < 0:
        trend = "Bearish"
    else:
        trend = "Sideways"
        
    # 2. Volatility Regime
    vol = row.get("daily_volatility_20d", 0.02) * 100.0
    if vol < 2.0:
        volatility = "Low"
    elif vol <= 3.5:
        volatility = "Medium"
    else:
        volatility = "High"
        
    # 3. Sentiment Regime
    news_sent = row.get("avg_news_sentiment", 0.0)
    tx_sent = row.get("transcript_sentiment", 0.0)
    # Combine (weighted average or whatever is valid)
    valid_sents = [s for s in [news_sent, tx_sent] if not pd.isna(s) and s != 0.0]
    avg_sent = np.mean(valid_sents) if valid_sents else 0.0
    
    if avg_sent > 0.10:
        sentiment = "Positive"
    elif avg_sent < -0.10:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return {
        "trend": trend,
        "volatility": volatility,
        "sentiment": sentiment,
        "raw_vol": vol,
        "raw_sent": avg_sent
    }

def get_signal_explainability(
    features_df: pd.DataFrame,
    ticker: str,
    date: str,
    feature_importance_df: pd.DataFrame,
    signal_type: int
) -> list:
    """
    Computes the top 5 contributing features for the current signal.
    """
    if features_df.empty or ticker not in features_df.index.get_level_values("ticker"):
        return []
        
    df_stock = features_df.xs(ticker, level="ticker")
    if date not in df_stock.index:
        return []
    row = df_stock.loc[date]
    
    # Get feature list to evaluate
    ignore_cols = ["open", "high", "low", "close", "adj_close", "volume", "label", "future_return_5d"]
    features = [c for c in features_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(features_df[c])]
    
    # Standardize and calculate influence
    importance_dict = {}
    if not feature_importance_df.empty:
        importance_dict = dict(zip(feature_importance_df["feature"], feature_importance_df["mean_importance"]))
        
    contributions = []
    for feat in features:
        raw_val = row[feat]
        if pd.isna(raw_val):
            continue
            
        mean_val = df_stock[feat].mean()
        std_val = df_stock[feat].std()
        
        # Calculate z-score
        z_score = (raw_val - mean_val) / std_val if std_val > 0.0 else 0.0
        importance = importance_dict.get(feat, 0.01)  # Default small importance if missing
        
        # Base influence score
        influence = z_score * importance
        
        # Determine if it supports the signal
        if signal_type == 1:  # BUY (2)
            # Higher values for positive indicators, lower values for negative indicators support BUY
            # To simplify, if the sign of z_score aligns with expected direction
            # We can classify based on whether z_score * importance is positive or negative
            is_supporting = (influence > 0)
            if "rsi" in feat:  # RSI low is oversold, which is bullish
                is_supporting = (z_score < 0)
        elif signal_type == -1:  # SELL (0)
            is_supporting = (influence < 0)
            if "rsi" in feat:  # RSI high is overbought, which is bearish
                is_supporting = (z_score > 0)
        else:  # HOLD (1)
            # For HOLD, supporting means close to mean (small absolute z-score)
            is_supporting = (abs(z_score) < 1.0)
            
        contributions.append({
            "feature": feat,
            "raw_val": raw_val,
            "z_score": z_score,
            "influence": abs(influence),
            "status": "Supporting" if is_supporting else "Opposing"
        })
        
    # Sort contributions by absolute influence descending
    df_contrib = pd.DataFrame(contributions)
    if df_contrib.empty:
        return []
        
    df_contrib = df_contrib.sort_values(by="influence", ascending=False)
    top_5 = df_contrib.head(5).to_dict(orient="records")
    return top_5

def get_live_signal_for_ticker(
    selected_ticker: str,
    quote: dict,
    features_df: pd.DataFrame,
    model_payload: dict
) -> dict:
    """
    In-memory merges the live quote with historical features, recomputes technicals,
    and runs model predictions safely without mutating features.parquet.
    """
    if features_df.empty or (selected_ticker not in features_df.index.get_level_values("ticker")):
        return {}
        
    # 1. Slice historical ticker data
    df_stock = features_df.xs(selected_ticker, level="ticker").copy()
    df_stock = df_stock.sort_index().reset_index()
    
    # 2. Append today's quote row
    from datetime import datetime
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    new_row = {
        "ticker": selected_ticker,
        "date": today_str,
        "open": quote["open"],
        "high": quote["high"],
        "low": quote["low"],
        "close": quote["current_price"],
        "adj_close": quote["current_price"],
        "volume": quote["volume"],
    }
    
    # Forward fill sentiment/macro metrics
    last_row = df_stock.iloc[-1]
    for col in df_stock.columns:
        if col not in new_row:
            new_row[col] = last_row[col]
            
    # Attempt to load today's actual news aggregates from the database
    try:
        from src.utils.db import get_db_connection
        conn = get_db_connection()
        agg_row = conn.execute("SELECT * FROM daily_aggregated_news WHERE ticker = ? AND date = ?", (selected_ticker, today_str)).fetchone()
        conn.close()
        if agg_row:
            new_row["avg_news_sentiment"] = agg_row["avg_news_sentiment"]
            new_row["news_sentiment_std"] = agg_row["news_sentiment_std"]
            new_row["news_count"] = agg_row["news_count"]
            new_row["avg_sentiment_3d"] = agg_row["avg_sentiment_3d"]
            new_row["news_count_3d"] = agg_row["news_count_3d"]
            new_row["pos_ratio_3d"] = agg_row["pos_ratio_3d"]
            new_row["neg_ratio_3d"] = agg_row["neg_ratio_3d"]
    except Exception as e:
        logger.error(f"Error querying daily aggregated news for live signal: {e}")
            
    if today_str in df_stock["date"].values:
        idx = df_stock[df_stock["date"] == today_str].index[0]
        for k, v in new_row.items():
            df_stock.at[idx, k] = v
    else:
        df_new = pd.DataFrame([new_row])
        df_stock = pd.concat([df_stock, df_new], ignore_index=True)
        
    # 3. Recompute technical indicators
    from src.features.technicals import compute_ticker_technicals
    df_stock = compute_ticker_technicals(df_stock)
    
    # 4. Prepare feature columns
    import json
    feature_cols_path = r"c:\Users\Agrim Sharma\Desktop\StockSig\reports\feature_columns.json"
    with open(feature_cols_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
        
    # 5. Model prediction on the latest row
    latest_row = df_stock.iloc[[-1]]
    X = latest_row[feature_cols].copy()
    
    model = model_payload['model']
    scaler = model_payload['scaler']
    
    if scaler is not None:
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)
    else:
        probs = model.predict_proba(X.values)
        
    prob_sell = float(probs[0, 0])
    prob_hold = float(probs[0, 1])
    prob_buy = float(probs[0, 2])
    
    # 6. Apply confirmations on the latest row
    from src.signals.filters import check_buy_confirmations, check_sell_confirmations
    buy_confirmed = bool(check_buy_confirmations(df_stock).iloc[-1])
    sell_confirmed = bool(check_sell_confirmations(df_stock).iloc[-1])
    
    # Threshold checks
    buy_threshold = 0.50
    sell_threshold = 0.55
    
    # Determine raw signal
    raw_sig = 1 # HOLD
    if prob_buy > buy_threshold:
        raw_sig = 2 # BUY
    elif prob_sell > sell_threshold:
        raw_sig = 0 # SELL
        
    # Apply confirmations to get final signal
    final_sig = raw_sig
    if raw_sig == 2 and not buy_confirmed:
        final_sig = 1
    elif raw_sig == 0 and not sell_confirmed:
        final_sig = 1
        
    # Map back: SELL=0->-1, HOLD=1->0, BUY=2->1
    signal_val = 0
    if final_sig == 2:
        signal_val = 1
    elif final_sig == 0:
        signal_val = -1
        
    # Confidence score calculation
    if final_sig == 2:
        conf = prob_buy
    elif final_sig == 0:
        conf = prob_sell
    else:
        conf = prob_hold
        
    label_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
    
    # Structure df_stock back to multi-indexed features DataFrame
    df_stock["ticker"] = selected_ticker
    df_stock_live_indexed = df_stock.set_index(["ticker", "date"])
    
    # Construct signals_df style DataFrame for the live date
    sig_row = pd.DataFrame([{
        "ticker": selected_ticker,
        "date": today_str,
        "signal": signal_val,
        "confidence_score": float(conf * 100.0),
        "prob_sell": prob_sell,
        "prob_hold": prob_hold,
        "prob_buy": prob_buy
    }]).set_index(["ticker", "date"])
    
    return {
        "label": label_map[signal_val],
        "confidence_score": float(conf * 100.0),
        "prob_buy": prob_buy,
        "prob_hold": prob_hold,
        "prob_sell": prob_sell,
        "df_stock_live": df_stock_live_indexed,
        "sig_live_df": sig_row,
        "date": today_str
    }

