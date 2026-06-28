import os
import pickle
import pandas as pd
import json
import logging
import streamlit as st

logger = logging.getLogger(__name__)

# Resolve project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_resolved_path(primary_rel_path: str, fallback_filename: str) -> str:
    """
    Checks if primary file exists. If not, falls back to the demo folder.
    """
    primary_path = os.path.join(BASE_DIR, *primary_rel_path.split("/"))
    if os.path.exists(primary_path):
        return primary_path
    
    # Fallback to data/demo/filename
    fallback_path = os.path.join(BASE_DIR, "data", "demo", fallback_filename)
    if os.path.exists(fallback_path):
        return fallback_path
        
    return primary_path

MODEL_PATH = get_resolved_path("models/calibrated_best.pkl", "calibrated_best.pkl")
FEATURES_PATH = get_resolved_path("data/features/features.parquet", "features.parquet")
SIGNALS_PATH = get_resolved_path("data/signals/signals.parquet", "signals.parquet")
EQUITY_CURVE_PATH = get_resolved_path("reports/equity_curve.parquet", "equity_curve.parquet")
TRADE_LOG_PATH = get_resolved_path("reports/trade_log.csv", "trade_log.csv")
FEAT_IMPORTANCE_PATH = get_resolved_path("reports/feature_importance.csv", "feature_importance.csv")
BACKTEST_REPORT_PATH = get_resolved_path("reports/backtest_report.json", "backtest_report.json")
BENCHMARK_PATH = get_resolved_path("reports/benchmark_comparison.json", "benchmark_comparison.json")

@st.cache_resource
def load_model():
    """Loads the calibrated best classifier model from disk."""
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        logger.info("Successfully loaded calibrated model.")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_features() -> pd.DataFrame:
    """Loads features.parquet into a DataFrame with standardized index."""
    if not os.path.exists(FEATURES_PATH):
        logger.error(f"Features file not found at {FEATURES_PATH}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(FEATURES_PATH)
        # Ensure date index level is string
        if "date" in df.index.names:
            df.index = df.index.set_levels(df.index.levels[1].astype(str), level="date")
        return df
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return pd.DataFrame()

@st.cache_data
def load_signals() -> pd.DataFrame:
    """Loads signals.parquet into a DataFrame with standardized index."""
    if not os.path.exists(SIGNALS_PATH):
        logger.error(f"Signals file not found at {SIGNALS_PATH}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(SIGNALS_PATH)
        if "date" in df.index.names:
            df.index = df.index.set_levels(df.index.levels[1].astype(str), level="date")
        return df
    except Exception as e:
        logger.error(f"Error loading signals: {e}")
        return pd.DataFrame()

@st.cache_data
def load_equity_curve() -> pd.DataFrame:
    """Loads equity_curve.parquet into a DataFrame."""
    if not os.path.exists(EQUITY_CURVE_PATH):
        logger.error(f"Equity curve file not found at {EQUITY_CURVE_PATH}")
        return pd.DataFrame()
    try:
        df = pd.read_parquet(EQUITY_CURVE_PATH)
        df["date"] = df["date"].astype(str)
        return df
    except Exception as e:
        logger.error(f"Error loading equity curve: {e}")
        return pd.DataFrame()

@st.cache_data
def load_trade_log() -> pd.DataFrame:
    """Loads trade_log.csv into a DataFrame."""
    if not os.path.exists(TRADE_LOG_PATH):
        logger.error(f"Trade log file not found at {TRADE_LOG_PATH}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(TRADE_LOG_PATH)
        df["date"] = df["date"].astype(str)
        if "entry_date" in df.columns:
            df["entry_date"] = df["entry_date"].astype(str)
        if "exit_date" in df.columns:
            df["exit_date"] = df["exit_date"].astype(str)
        return df
    except Exception as e:
        logger.error(f"Error loading trade log: {e}")
        return pd.DataFrame()

@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Loads feature_importance.csv into a DataFrame."""
    if not os.path.exists(FEAT_IMPORTANCE_PATH):
        logger.error(f"Feature importance file not found at {FEAT_IMPORTANCE_PATH}")
        # Fallback: look in models or create mock
        alt_path = os.path.join(BASE_DIR, "reports", "feature_importance.csv")
        if os.path.exists(alt_path):
            try:
                return pd.read_csv(alt_path)
            except:
                pass
        return pd.DataFrame(columns=["feature", "xgboost_importance", "lightgbm_importance", "mean_importance"])
    try:
        return pd.read_csv(FEAT_IMPORTANCE_PATH)
    except Exception as e:
        logger.error(f"Error loading feature importance: {e}")
        return pd.DataFrame()

@st.cache_data
def load_backtest_report() -> dict:
    """Loads backtest_report.json as a dictionary."""
    if not os.path.exists(BACKTEST_REPORT_PATH):
        logger.error(f"Backtest report file not found at {BACKTEST_REPORT_PATH}")
        return {}
    try:
        with open(BACKTEST_REPORT_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading backtest report: {e}")
        return {}

@st.cache_data
def load_benchmark_comparison() -> dict:
    """Loads benchmark_comparison.json as a dictionary."""
    if not os.path.exists(BENCHMARK_PATH):
        logger.error(f"Benchmark comparison file not found at {BENCHMARK_PATH}")
        return {}
    try:
        with open(BENCHMARK_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading benchmark comparison: {e}")
        return {}
