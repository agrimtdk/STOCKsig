import os
import sys
import logging
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.backtest.backtester import run_backtest_pipeline

# Configure Logging
reports_dir = r"c:\Users\Agrim Sharma\Desktop\StockSig\reports"
logs_dir = os.path.join(reports_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(logs_dir, "phase5.log"), mode="w", encoding="utf-8")
    ]
)

logger = logging.getLogger("run_phase5")

def main():
    signals_path = r"c:\Users\Agrim Sharma\Desktop\StockSig\data\signals\signals.parquet"
    features_path = r"c:\Users\Agrim Sharma\Desktop\StockSig\data\features\features.parquet"
    
    logger.info("Starting Phase 5 Backtesting Run...")
    
    try:
        report = run_backtest_pipeline(
            signals_path=signals_path,
            features_path=features_path,
            reports_dir=reports_dir,
            initial_capital=100000.0
        )
        
        # Extract metrics for printing
        sm = report["strategy_metrics"]
        bc = report["benchmark_comparison"]
        bm = bc["benchmark"]
        comp = bc["comparison"]
        
        print("\n" + "=" * 50)
        print(" PHASE 5 - BACKTESTING SUMMARY RESULTS ")
        print("=" * 50)
        print(f"Strategy Period:  {bc['start_date']} to {bc['end_date']}")
        print(f"Benchmark Used:   {bc['benchmark_name']}")
        print("-" * 50)
        print(f"1. Total Return:      {sm['total_return']:.2%}")
        print(f"2. CAGR:              {sm['cagr']:.2%}")
        print(f"3. Sharpe Ratio:      {sm['sharpe_ratio']:.2f}")
        print(f"4. Sortino Ratio:     {sm['sortino_ratio']:.2f}")
        print(f"5. Max Drawdown:      {sm['max_drawdown']:.2%}")
        print(f"6. Win Rate:          {sm['win_rate']:.2%}")
        print(f"7. Total Trades:      {sm['total_trades']}")
        print(f"8. Benchmark Return:  {bm['total_return']:.2%}")
        print(f"9. Alpha vs Bench:    {comp['alpha_return']:.2%}")
        print("-" * 50)
        print(" CRITICAL VALIDATION DIAGNOSTICS ")
        print("-" * 50)
        print(f"Average Holding Period (Days):      {sm['average_holding_period']:.2f} calendar days")
        print(f"Signal-to-Trade Conversion Rate:     {sm['signal_to_trade_conversion_rate']:.2%}")
        print(f"Mean Exposure Utilization %:         {sm['mean_exposure_utilization']:.2%}")
        print("=" * 50 + "\n")
        
        logger.info("Phase 5 Backtesting completed successfully.")
        
    except Exception as e:
        logger.error(f"Backtesting run failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
