import logging

logger = logging.getLogger(__name__)

GLOSSARY = {
    "MACD": "Measures trend momentum using moving averages.\nPositive = bullish momentum.\nNegative = weakening trend.",
    "RSI": "Shows if a stock is overbought or oversold.\nAbove 70 may mean expensive (overbought).\nBelow 30 may mean cheap (oversold).",
    "ATR": "Average True Range. Measures market volatility.\nHigher values indicate larger daily price swings.",
    "OBV": "On-Balance Volume. Relates price changes to volume.\nRising OBV shows buying pressure on up days.",
    "Bollinger Bands": "Volatility bands around a moving average.\nBands expand during high volatility and contract during low.",
    "Volatility": "Measures the speed and magnitude of price changes.\nHigh volatility means higher risk and wider price ranges.",
    "Sharpe Ratio": "Measures return earned per unit of total risk.\nHigher is better; above 1.0 is considered good.",
    "Sortino Ratio": "Measures return per unit of downside risk only.\nHigher is better; ignores beneficial upside volatility.",
    "Max Drawdown": "The peak-to-trough decline during a specific period.\nMeasures the worst-case potential loss of a strategy.",
    "CAGR": "Compound Annual Growth Rate.\nThe smoothed annual rate of return over a multi-year period.",
    "Alpha": "Measures strategy outperformance relative to a benchmark.\nPositive Alpha means the strategy beat the benchmark.",
    "Beta": "Measures strategy sensitivity to market movements.\nBeta of 1.0 moves with the market; >1.0 is more volatile.",
    "Win Rate": "The percentage of total trades that ended in a profit.\nWin Rate = Profitable Trades / Total Trades.",
    "Profit Factor": "Gross profits divided by gross losses.\nValues above 1.0 indicate a profitable strategy.",
    "Expectancy": "The average amount expected to win or lose per trade.\nPositive expectancy means a profitable system over time.",
    "Signal Confidence": "Model probability score for the active classification.\nHigher percentage indicates stronger statistical agreement.",
    "Market Regime": "Categorizes the current state of the market (e.g. Bullish).\nUsed to align active signal execution filters.",
    "Sentiment Score": "Weighted sentiment classification from news/transcripts.\nPositive values mean bullish news coverage.",
    "Moving Average": "Smoothed average price over a specified number of days.\nHelps identify the underlying trend direction.",
    "EMA": "Exponential Moving Average. Gives more weight to recent prices.\nReacts faster to recent price changes than a simple average.",
    "SMA": "Simple Moving Average. Arithmetic mean of past prices.\nActs as a baseline support/resistance line.",
    "Volume Z-score": "Measures how current volume compares to its historical average.\nValues > 2.0 indicate unusually high trading activity.",
    "Trend": "Indicates the general direction of price movement (upward, downward, or sideways) over time.\nBullish = upward trend; Bearish = downward trend."
}

# Alias/Minor variations map
ALIASES = {
    "SHARPE": "Sharpe Ratio",
    "SORTINO": "Sortino Ratio",
    "MAX_DRAWDOWN": "Max Drawdown",
    "BOLLINGER": "Bollinger Bands",
    "VOLUME_ZSCORE": "Volume Z-score",
    "SENTIMENT": "Sentiment Score",
    "CONFIDENCE": "Signal Confidence",
    "REGIME": "Market Regime"
}

def tooltip(term: str, label: str = None) -> str:
    """
    Returns an HTML span representing a glossary tooltip element.
    Uses glassmorphism styles on hover.
    """
    cleaned_term = term.strip().upper()
    
    # Try direct match or alias lookup
    matched_term = None
    if term in GLOSSARY:
        matched_term = term
    elif cleaned_term in GLOSSARY:
        matched_term = cleaned_term
    elif cleaned_term in ALIASES:
        matched_term = ALIASES[cleaned_term]
    else:
        # Case insensitive check
        for k in GLOSSARY.keys():
            if k.upper() == cleaned_term:
                matched_term = k
                break
                
    if not matched_term:
        logger.warning(f"Glossary term not found: {term}")
        display_label = label if label is not None else term
        return display_label

    definition = GLOSSARY[matched_term]
    display_label = label if label is not None else matched_term
    
    # Replace newlines with <br/> for HTML rendering
    definition_html = definition.replace("\n", "<br/>")
    
    html = (
        f'<span class="glossary-tooltip">'
        f'{display_label}'
        f'<span class="glossary-tooltiptext">{definition_html}</span>'
        f'</span>'
    )
    return html
