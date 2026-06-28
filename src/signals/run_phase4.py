import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import config
from src.signals.threshold_optimizer import optimize_thresholds
from src.signals.signal_generator import generate_signals

def setup_logging():
    """Configures global python logging to console and to a local log file."""
    log_file = config.prices_dir.parent.parent.parent / "reports" / "logs" / "phase4.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file), encoding="utf-8")
        ]
    )

def main():
    setup_logging()
    logger = logging.getLogger("run_phase4")
    
    logger.info("==================================================")
    logger.info("Starting Phase 4 Signal Engine Pipeline")
    logger.info("==================================================")
    
    try:
        # 1. Threshold Optimization
        buy_thresh, sell_thresh = optimize_thresholds()
        
        # 2. Signal Generation and Validation Filtering
        signals_df, stats = generate_signals(
            buy_threshold=buy_thresh,
            sell_threshold=sell_thresh
        )
        
        # 3. Save Parquet
        signals_dir = config.prices_dir.parent.parent / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        signals_parquet = signals_dir / "signals.parquet"
        signals_df.to_parquet(signals_parquet)
        logger.info(f"Saved signals dataset to {signals_parquet}")
        
        # 4. Generate Signal Report stats
        logger.info("Compiling signal generation report...")
        total_rows = len(signals_df)
        
        # Signal distribution
        dist = signals_df['signal'].value_counts().to_dict()
        dist_str = {
            "BUY (1)": int(dist.get(1, 0)),
            "SELL (-1)": int(dist.get(-1, 0)),
            "HOLD (0)": int(dist.get(0, 0))
        }
        
        # Avg confidence by class
        avg_conf = signals_df.groupby('signal')['confidence_score'].mean().to_dict()
        avg_conf_str = {
            "BUY (1)": float(avg_conf.get(1, 0.0)),
            "SELL (-1)": float(avg_conf.get(-1, 0.0)),
            "HOLD (0)": float(avg_conf.get(0, 0.0))
        }
        
        # Signal frequency by ticker
        ticker_freq = {}
        for ticker, grp in signals_df.reset_index().groupby('ticker'):
            trades = np.sum(grp['signal'] != 0)
            ticker_freq[ticker] = float(trades / len(grp))
            
        report = {
            "best_buy_threshold": buy_thresh,
            "best_sell_threshold": sell_thresh,
            "final_buy_threshold": stats.get("final_buy_threshold", buy_thresh),
            "final_sell_threshold": stats.get("final_sell_threshold", sell_thresh),
            "fallback_triggered": stats.get("fallback_triggered", {"BUY": False, "SELL": False}),
            "rejections": stats['rejections'],
            "raw_signals": stats['raw_signals'],
            "confirmed_signals": stats['confirmed_signals'],
            "signal_distribution": dist_str,
            "average_confidence": avg_conf_str,
            "ticker_signal_frequency": ticker_freq
        }
        
        report_path = config.prices_dir.parent.parent.parent / "reports" / "signal_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Saved signal report to {report_path}")
        
        # Load threshold report to print OOF metrics
        thresh_report_path = config.prices_dir.parent.parent.parent / "reports" / "threshold_report.json"
        with open(thresh_report_path, 'r', encoding='utf-8') as f:
            t_rep = json.load(f)

        # --- PRINT REQUIRED OUTPUT SUMMARY ---
        print("\n" + "="*50)
        print("PHASE 4 - SIGNAL ENGINE SUMMARY")
        print("="*50)
        
        # 1. Optimized Thresholds
        print(f"\n[1] Optimized Probabilistic Thresholds:")
        print(f"  Optimized BUY Threshold:  {buy_thresh:.2f}")
        print(f"  Optimized SELL Threshold: {sell_thresh:.2f}")
        
        # 2. Signal Distribution
        print(f"\n[2] Confirmed Signal Distribution:")
        print(f"  BUY (1) Trades:  {dist_str['BUY (1)']} rows ({dist_str['BUY (1)']/total_rows*100:.2f}%)")
        print(f"  SELL (-1) Trades: {dist_str['SELL (-1)']} rows ({dist_str['SELL (-1)']/total_rows*100:.2f}%)")
        print(f"  HOLD (0) States:  {dist_str['HOLD (0)']} rows ({dist_str['HOLD (0)']/total_rows*100:.2f}%)")
        
        # 3. Average Confidence
        print(f"\n[3] Average Confidence Score (0-100%):")
        print(f"  BUY Confidence:  {avg_conf_str['BUY (1)']:.2f}%")
        print(f"  SELL Confidence: {avg_conf_str['SELL (-1)']:.2f}%")
        print(f"  HOLD Confidence: {avg_conf_str['HOLD (0)']:.2f}%")
        
        # 4. Filter Rejection %
        print(f"\n[4] Technical Filters Rejection Rates:")
        rej = stats['rejections']
        print(f"  BUY Raw Signals Rejected:  {rej['BUY_rejected']} / {stats['raw_signals']['BUY']} ({rej['BUY_rejection_rate']*100:.2f}%)")
        print(f"  SELL Raw Signals Rejected: {rej['SELL_rejected']} / {stats['raw_signals']['SELL']} ({rej['SELL_rejection_rate']*100:.2f}%)")
        print(f"  Total Signals Rejected:    {rej['BUY_rejected'] + rej['SELL_rejected']} / {stats['raw_signals']['BUY'] + stats['raw_signals']['SELL']} ({rej['total_rejection_rate']*100:.2f}%)")
        
        # 5. Fallback Status
        print(f"\n[5] Fallback Activation & Final Thresholds:")
        print(f"  BUY Fallback Triggered:  {report['fallback_triggered']['BUY']}")
        print(f"  SELL Fallback Triggered: {report['fallback_triggered']['SELL']}")
        print(f"  Final BUY Threshold Used:  {report['final_buy_threshold']:.4f}")
        print(f"  Final SELL Threshold Used: {report['final_sell_threshold']:.4f}")
        
        # 6. OOF Metrics
        print(f"\n[6] Out-of-Fold Class Quality Metrics:")
        print(f"  BUY Class  - Precision: {t_rep.get('oof_buy_precision', 0.0):.4f}, Recall: {t_rep.get('oof_buy_recall', 0.0):.4f}")
        print(f"  SELL Class - Precision: {t_rep.get('oof_sell_precision', 0.0):.4f}, Recall: {t_rep.get('oof_sell_recall', 0.0):.4f}")
        
        # Reset index to query ticker/date
        df_reset = signals_df.reset_index()
        
        # 7. Top 5 highest-confidence BUYs
        if dist_str['BUY (1)'] > 0:
            print(f"\n[7] Top 5 Highest-Confidence BUY Signals:")
            buys_sorted = df_reset[df_reset['signal'] == 1].sort_values(by='confidence_score', ascending=False)
            print(buys_sorted[['ticker', 'date', 'confidence_score', 'prob_buy']].head(5).to_string(index=False))
        else:
            print(f"\n[7] Top 5 Highest-Confidence BUY Signals: None generated")
            
        # 8. Top 5 highest-confidence SELLs
        if dist_str['SELL (-1)'] > 0:
            print(f"\n[8] Top 5 Highest-Confidence SELL Signals:")
            sells_sorted = df_reset[df_reset['signal'] == -1].sort_values(by='confidence_score', ascending=False)
            print(sells_sorted[['ticker', 'date', 'confidence_score', 'prob_sell']].head(5).to_string(index=False))
        else:
            print(f"\n[8] Top 5 Highest-Confidence SELL Signals: None generated")
        
        print("\n" + "="*50)
        logger.info("Phase 4 pipeline execution completed successfully.")
        
    except Exception as e:
        logger.error(f"Phase 4 execution failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
