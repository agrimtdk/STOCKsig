import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_volatility_risk_level(volatility_percent: float) -> str:
    """
    Returns the risk level (LOW, MEDIUM, HIGH) based on the stock's volatility.
    """
    if pd.isna(volatility_percent):
        return "UNKNOWN"
        
    if volatility_percent < 2.0:
        return "LOW"
    elif volatility_percent <= 3.5:
        return "MEDIUM"
    else:
        return "HIGH"

def format_metric_value(value: float, metric_type: str) -> str:
    """
    Formats numeric metrics as clean, readable strings.
    """
    if pd.isna(value):
        return "N/A"
        
    if metric_type == "percent":
        return f"{value * 100:.2f}%"
    elif metric_type == "percent_raw":
        # Already scaled to 100
        return f"{value:.2f}%"
    elif metric_type == "ratio":
        return f"{value:.2f}"
    elif metric_type == "count":
        return f"{int(value)}"
    elif metric_type == "currency":
        return f"{value:,.2f} INR"
    elif metric_type == "days":
        return f"{value:.1f} Days"
    else:
        return f"{value}"
