import json
import logging
import numpy as np
import pandas as pd
from typing import Tuple
from pathlib import Path
from sklearn.metrics import precision_score, recall_score
from src.utils.config import config

logger = logging.getLogger(__name__)

def optimize_thresholds(oof_path: Path = None) -> Tuple[float, float]:
    """
    Grid searches BUY/SELL probability thresholds to maximize validation F1_signal
    subject to minimum/maximum signal frequency constraints (2% <= freq <= 20% for each).
    """
    if oof_path is None:
        oof_path = config.prices_dir.parent.parent / "predictions" / "oof_predictions.parquet"
        
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF predictions file not found at {oof_path}. Run Phase 3 first.")
        
    logger.info(f"Loading OOF validation predictions from {oof_path}...")
    df = pd.read_parquet(oof_path)
    
    # Load calibrated model and calibrate OOF predictions to align probability spaces
    model_path = config.prices_dir.parent.parent.parent / "models" / "calibrated_best.pkl"
    if model_path.exists():
        logger.info(f"Loading calibrated model from {model_path} to align OOF probability space...")
        import pickle
        with open(model_path, 'rb') as f:
            payload = pickle.load(f)
        calibrated_clf = payload['model']
        if hasattr(calibrated_clf, "calibrated_classifiers_"):
            prob_sell = np.mean([clf.calibrators[0].predict(df['prob_sell'].values) for clf in calibrated_clf.calibrated_classifiers_], axis=0)
            prob_buy = np.mean([clf.calibrators[2].predict(df['prob_buy'].values) for clf in calibrated_clf.calibrated_classifiers_], axis=0)
        else:
            logger.info("Fitted model is not CalibratedClassifierCV. Using uncalibrated OOF predictions.")
            prob_sell = df['prob_sell'].values
            prob_buy = df['prob_buy'].values
    else:
        logger.warning(f"Calibrated best model not found at {model_path}. Using uncalibrated OOF predictions.")
        prob_sell = df['prob_sell'].values
        prob_buy = df['prob_buy'].values

    y_true = df['true_label'].values.astype(int)
    
    # Search grid
    thresholds = np.arange(0.35, 0.71, 0.05)
    
    best_buy_thresh = 0.50
    best_sell_thresh = 0.50
    best_score = -1.0
    best_freq = 0.0
    best_buy_freq = 0.0
    best_sell_freq = 0.0
    best_buy_prec = 0.0
    best_sell_prec = 0.0
    best_buy_rec = 0.0
    best_sell_rec = 0.0
    
    logger.info("Starting threshold optimization grid search...")
    
    for t_buy in thresholds:
        for t_sell in thresholds:
            # Predict labels based on thresholds
            preds = np.ones(len(df), dtype=int) # Default is HOLD (1)
            
            is_buy = prob_buy > t_buy
            is_sell = prob_sell > t_sell
            
            # Resolve overlaps by max probability
            overlap = is_buy & is_sell
            preds[is_buy] = 2 # BUY
            preds[is_sell] = 0 # SELL
            
            if overlap.any():
                preds[overlap] = np.where(prob_buy[overlap] > prob_sell[overlap], 2, 0)
                
            # Compute signal frequencies
            buy_count = np.sum(preds == 2)
            sell_count = np.sum(preds == 0)
            buy_freq = buy_count / len(preds)
            sell_freq = sell_count / len(preds)
            total_freq = (buy_count + sell_count) / len(preds)
            
            # Skip if frequency constraints (2% <= freq <= 20% for both) are violated
            if not (0.02 <= buy_freq <= 0.20) or not (0.02 <= sell_freq <= 0.20):
                continue
                
            # Align thresholds to meet target ranges on features (BUY: 5%-8% on OOF, SELL: 8%-15% on OOF)
            if not (0.05 <= buy_freq <= 0.08) or not (0.08 <= sell_freq <= 0.15):
                continue
                
            # Calculate BUY precision & recall
            buy_prec = precision_score(y_true == 2, preds == 2, zero_division=0)
            buy_rec = recall_score(y_true == 2, preds == 2, zero_division=0)
            buy_score = 0.6 * buy_prec + 0.4 * buy_rec
                
            # Calculate SELL precision & recall
            sell_prec = precision_score(y_true == 0, preds == 0, zero_division=0)
            sell_rec = recall_score(y_true == 0, preds == 0, zero_division=0)
            sell_score = 0.6 * sell_prec + 0.4 * sell_rec
                
            score = buy_score + sell_score
            
            # Look for max score
            if score > best_score:
                best_score = score
                best_buy_thresh = float(t_buy)
                best_sell_thresh = float(t_sell)
                best_freq = float(total_freq)
                best_buy_freq = float(buy_freq)
                best_sell_freq = float(sell_freq)
                best_buy_prec = float(buy_prec)
                best_sell_prec = float(sell_prec)
                best_buy_rec = float(buy_rec)
                best_sell_rec = float(sell_rec)

    # If no threshold satisfied the constraint, fall back to 0.50
    if best_score == -1.0:
        logger.warning("No threshold combination satisfied the constraints. Falling back to default 0.50.")
        best_buy_thresh = 0.50
        best_sell_thresh = 0.50
        # Calculate OOF stats at default 0.50
        preds_def = np.ones(len(df), dtype=int)
        is_buy_def = prob_buy > 0.50
        is_sell_def = prob_sell > 0.50
        preds_def[is_buy_def] = 2
        preds_def[is_sell_def] = 0
        overlap_def = is_buy_def & is_sell_def
        if overlap_def.any():
            preds_def[overlap_def] = np.where(prob_buy[overlap_def] > prob_sell[overlap_def], 2, 0)
        best_freq = float(np.sum(preds_def != 1) / len(df))
        best_buy_freq = float(np.sum(preds_def == 2) / len(df))
        best_sell_freq = float(np.sum(preds_def == 0) / len(df))
        best_buy_prec = float(precision_score(y_true == 2, preds_def == 2, zero_division=0))
        best_sell_prec = float(precision_score(y_true == 0, preds_def == 0, zero_division=0))
        best_buy_rec = float(recall_score(y_true == 2, preds_def == 2, zero_division=0))
        best_sell_rec = float(recall_score(y_true == 0, preds_def == 0, zero_division=0))
        
    logger.info(f"Optimization complete. Selected Thresholds:")
    logger.info(f"  BUY Threshold:  {best_buy_thresh:.2f} (OOF Prec: {best_buy_prec:.4f}, Rec: {best_buy_rec:.4f})")
    logger.info(f"  SELL Threshold: {best_sell_thresh:.2f} (OOF Prec: {best_sell_prec:.4f}, Rec: {best_sell_rec:.4f})")
    logger.info(f"  Combined Precision/Recall Score: {best_score:.4f}")
    logger.info(f"  Signal Frequency: {best_freq*100:.2f}% (BUY: {best_buy_freq*100:.2f}%, SELL: {best_sell_freq*100:.2f}%)")
    
    # Save parameters to report json
    report_path = config.prices_dir.parent.parent.parent / "reports" / "threshold_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "best_buy_threshold": best_buy_thresh,
        "best_sell_threshold": best_sell_thresh,
        "oof_buy_precision": best_buy_prec,
        "oof_buy_recall": best_buy_rec,
        "oof_sell_precision": best_sell_prec,
        "oof_sell_recall": best_sell_rec,
        "combined_score": best_score,
        "signal_frequency": best_freq,
        "buy_frequency": best_buy_freq,
        "sell_frequency": best_sell_freq
    }
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"Saved threshold report to {report_path}")
    
    return best_buy_thresh, best_sell_thresh
