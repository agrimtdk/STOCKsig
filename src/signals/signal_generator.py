import json
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from src.utils.config import config
from src.signals.filters import check_buy_confirmations, check_sell_confirmations

logger = logging.getLogger(__name__)

def generate_signals(
    features_path: Path = None,
    model_path: Path = None,
    buy_threshold: float = 0.50,
    sell_threshold: float = 0.50
) -> Tuple[pd.DataFrame, dict]:
    """
    Runs model inference on features.parquet, applies probability thresholds,
    filters signals technically, computes confidence scores, and constructs the signals dataset.
    """
    if features_path is None:
        features_path = config.prices_dir.parent.parent / "features" / "features.parquet"
    if model_path is None:
        model_path = config.prices_dir.parent.parent.parent / "models" / "calibrated_best.pkl"
        
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found at {features_path}. Run Phase 2 first.")
    if not model_path.exists():
        raise FileNotFoundError(f"Calibrated best model not found at {model_path}. Run Phase 3 first.")

    # 1. Load data and features list
    logger.info(f"Loading features dataset for signal generation from {features_path}...")
    df = pd.read_parquet(features_path)
    
    feature_cols_path = config.prices_dir.parent.parent.parent / "reports" / "feature_columns.json"
    with open(feature_cols_path, 'r', encoding='utf-8') as f:
        feature_cols = json.load(f)
        
    X = df[feature_cols].copy()
    
    # 2. Load model & scaler payload
    logger.info(f"Loading calibrated best model from {model_path}...")
    with open(model_path, 'rb') as f:
        payload = pickle.load(f)
    model = payload['model']
    scaler = payload['scaler']
    
    # 3. Predict calibrated probabilities
    if scaler is not None:
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)
    else:
        probs = model.predict_proba(X.values)
        
    prob_sell = probs[:, 0]
    prob_hold = probs[:, 1]
    prob_buy = probs[:, 2]
    
    # 5. Apply technical filters (confirmations)
    logger.info("Applying technical indicator confirmation filters...")
    buy_confirmed = check_buy_confirmations(df)
    sell_confirmed = check_sell_confirmations(df)
    
    fallback_triggered = {"BUY": False, "SELL": False}
    
    def compute_final_signals(b_thresh, s_thresh):
        raw = np.ones(len(df), dtype=int) # Default is HOLD (1)
        is_buy = prob_buy > b_thresh
        is_sell = prob_sell > s_thresh
        raw[is_buy] = 2 # BUY
        raw[is_sell] = 0 # SELL
        overlap = is_buy & is_sell
        if overlap.any():
            raw[overlap] = np.where(prob_buy[overlap] > prob_sell[overlap], 2, 0)
            
        final = raw.copy()
        final[(raw == 2) & (~buy_confirmed)] = 1 # Downgrade BUY
        final[(raw == 0) & (~sell_confirmed)] = 1 # Downgrade SELL
        return raw, final

    raw_signals, final_signals = compute_final_signals(buy_threshold, sell_threshold)
    
    # Check if final confirmed BUY count == 0
    if np.sum(final_signals == 2) == 0:
        logger.warning("Confirmed BUY signals count is 0 on features dataset. Triggering OOF fallback threshold.")
        fallback_triggered["BUY"] = True
        oof_path = config.prices_dir.parent.parent / "predictions" / "oof_predictions.parquet"
        if oof_path.exists():
            oof_df = pd.read_parquet(oof_path)
            buy_threshold = float(np.percentile(oof_df['prob_buy'].values, 75))
        else:
            logger.warning(f"OOF predictions file not found at {oof_path}. Falling back to in-sample 75th percentile for BUY.")
            buy_threshold = float(np.percentile(prob_buy, 75))
        logger.info(f"Forced BUY threshold reset to OOF 75th percentile: {buy_threshold:.4f}")
        raw_signals, final_signals = compute_final_signals(buy_threshold, sell_threshold)
        
    # Check if final confirmed SELL count == 0
    if np.sum(final_signals == 0) == 0:
        logger.warning("Confirmed SELL signals count is 0 on features dataset. Triggering OOF fallback threshold.")
        fallback_triggered["SELL"] = True
        oof_path = config.prices_dir.parent.parent / "predictions" / "oof_predictions.parquet"
        if oof_path.exists():
            oof_df = pd.read_parquet(oof_path)
            sell_threshold = float(np.percentile(oof_df['prob_sell'].values, 75))
        else:
            logger.warning(f"OOF predictions file not found at {oof_path}. Falling back to in-sample 75th percentile for SELL.")
            sell_threshold = float(np.percentile(prob_sell, 75))
        logger.info(f"Forced SELL threshold reset to OOF 75th percentile: {sell_threshold:.4f}")
        raw_signals, final_signals = compute_final_signals(buy_threshold, sell_threshold)
        
    # Re-calculate statistics based on final values
    raw_buy_count = int(np.sum(raw_signals == 2))
    raw_sell_count = int(np.sum(raw_signals == 0))
    raw_hold_count = int(np.sum(raw_signals == 1))
    
    buy_rejected_count = int(np.sum((raw_signals == 2) & (final_signals == 1)))
    sell_rejected_count = int(np.sum((raw_signals == 0) & (final_signals == 1)))
    
    buy_rejection_rate = buy_rejected_count / max(1, raw_buy_count)
    sell_rejection_rate = sell_rejected_count / max(1, raw_sell_count)
    total_rejection_rate = (buy_rejected_count + sell_rejected_count) / max(1, raw_buy_count + raw_sell_count)
    
    # 6. Calculate confidence score (normalized 0-100)
    confidence = np.zeros(len(df))
    confidence[final_signals == 2] = prob_buy[final_signals == 2]
    confidence[final_signals == 0] = prob_sell[final_signals == 0]
    confidence[final_signals == 1] = prob_hold[final_signals == 1]
    confidence_score = confidence * 100.0
    
    # 7. Map back to original labelling space: SELL=0->-1, HOLD=1->0, BUY=2->1
    signal_mapped = np.zeros(len(df), dtype=int)
    signal_mapped[final_signals == 2] = 1 # BUY
    signal_mapped[final_signals == 0] = -1 # SELL
    signal_mapped[final_signals == 1] = 0 # HOLD
    
    # 8. Create output DataFrame
    df_out = df.reset_index()[['ticker', 'date']].copy()
    df_out['signal'] = signal_mapped
    df_out['confidence_score'] = confidence_score
    df_out['prob_sell'] = prob_sell
    df_out['prob_hold'] = prob_hold
    df_out['prob_buy'] = prob_buy
    
    df_out = df_out.set_index(['ticker', 'date'])
    
    # Compile statistics metadata
    stats = {
        "final_buy_threshold": buy_threshold,
        "final_sell_threshold": sell_threshold,
        "fallback_triggered": fallback_triggered,
        "raw_signals": {
            "BUY": raw_buy_count,
            "SELL": raw_sell_count,
            "HOLD": raw_hold_count
        },
        "confirmed_signals": {
            "BUY": int(np.sum(signal_mapped == 1)),
            "SELL": int(np.sum(signal_mapped == -1)),
            "HOLD": int(np.sum(signal_mapped == 0))
        },
        "rejections": {
            "BUY_rejected": buy_rejected_count,
            "SELL_rejected": sell_rejected_count,
            "BUY_rejection_rate": buy_rejection_rate,
            "SELL_rejection_rate": sell_rejection_rate,
            "total_rejection_rate": total_rejection_rate
        }
    }
    
    return df_out, stats
