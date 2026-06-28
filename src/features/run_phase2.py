import os
import logging
import pandas as pd
from src.utils.config import config
from src.features.feature_builder import FeatureBuilder

# Force mock NLP to speed up the process and make it robust (doesn't require internet downloads of BERT)
os.environ["STOCK_SIG_MOCK_NLP"] = "1"

def setup_logging():
    """Configures global python logging to console and to a local log file."""
    log_file = config.prices_dir.parent.parent.parent / "reports" / "logs" / "phase2.log"
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
    logger = logging.getLogger("run_phase2")
    
    logger.info("==================================================")
    logger.info("Starting Phase 2 Feature Engineering Pipeline")
    logger.info("==================================================")
    
    try:
        # Initialize and build features
        builder = FeatureBuilder(use_finbert=False) # Fallback to VADER/Mock for ingestion speed
        df = builder.build_features()
        
        logger.info("==================================================")
        logger.info("Phase 2 Executed Successfully.")
        logger.info("==================================================")
        logger.info(f"Total rows in final dataset: {len(df)}")
        logger.info(f"Total features: {len(df.columns) - 2}") # excludes label and future return
        
        # Display class distribution
        label_counts = df['label'].value_counts(dropna=False)
        logger.info("Class Distribution in Final Dataset:")
        for key, val in label_counts.items():
            lbl = "NaN (Warm-down)" if pd.isna(key) else ("BUY" if key == 1 else ("SELL" if key == -1 else "HOLD"))
            logger.info(f"  {lbl}: {val} rows ({val/len(df)*100:.2f}%)")
            
        # Display sample output (head)
        sample_cols = [
            'close', 'sma_5', 'rsi_14', 'macd', 
            'avg_news_sentiment', 'transcript_sentiment', 
            'macro_usdinr', 'label'
        ]
        logger.info("Sample DataFrame Rows:")
        print(df[sample_cols].head(10))
        
    except Exception as e:
        logger.error(f"Phase 2 execution failed: {e}", exc_info=True)
        raise e

if __name__ == "__main__":
    main()
