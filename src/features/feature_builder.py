import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from src.utils.config import config
from src.ingestion.data_loader import DataLoader
from src.features.technicals import compute_technical_indicators
from src.features.sentiment_features import (
    SentimentExtractor,
    extract_news_sentiments,
    extract_transcript_sentiments,
    get_macro_features
)
from src.features.label_generator import generate_labels

logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self, use_finbert: bool = True):
        self.loader = DataLoader()
        self.extractor = SentimentExtractor(use_finbert=use_finbert)
        self.features_dir = config.prices_dir.parent.parent / "features"
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.output_parquet = self.features_dir / "features.parquet"
        self.report_json = config.prices_dir.parent.parent.parent / "reports" / "phase2_feature_report.json"
        self.report_json.parent.mkdir(parents=True, exist_ok=True)

    def build_features(self) -> pd.DataFrame:
        logger.info("Initializing Feature Engineering Pipeline...")
        
        # 1. Load historical prices and compute technical indicators
        tickers = config.tickers
        logger.info(f"Loading prices for universe: {tickers}")
        prices_df = self.loader.load_prices(tickers)
        if prices_df.empty:
            raise ValueError("No price records found in database. Ingest price data first.")
            
        tech_df = compute_technical_indicators(prices_df)
        
        # 2. Load news, extract daily sentiments
        logger.info("Loading news articles...")
        news_df = self.loader.load_news(tickers)
        if not news_df.empty:
            news_sentiment = extract_news_sentiments(news_df, self.extractor)
        else:
            logger.warning("No news articles found. Creating empty sentiment placeholders.")
            news_sentiment = pd.DataFrame(columns=['ticker', 'date', 'avg_news_sentiment', 'news_sentiment_std', 'news_count'])
            
        # 3. Load transcripts, extract sentiments
        logger.info("Loading earnings call transcripts...")
        transcripts_df = self.loader.load_transcripts(tickers)
        if not transcripts_df.empty:
            transcript_sentiment = extract_transcript_sentiments(transcripts_df)
        else:
            logger.warning("No transcripts found. Creating empty transcript placeholders.")
            transcript_sentiment = pd.DataFrame(columns=['ticker', 'date', 'transcript_sentiment', 'transcript_length', 'positive_word_ratio', 'negative_word_ratio'])

        # 4. Generate Target Labels
        logger.info("Generating target labels...")
        labeled_df = generate_labels(tech_df)

        # 5. Extract Macro Economic Features
        start_date = labeled_df['date'].min()
        end_date = labeled_df['date'].max()
        macro_df = get_macro_features(start_date, end_date)

        # --- Merging Pipeline ---
        logger.info("Merging technical indicators, sentiments, macro features, and labels...")
        
        # Merge news daily aggregates (left join, default to 0 sentiment, 0 count)
        merged_df = pd.merge(labeled_df, news_sentiment, on=['ticker', 'date'], how='left')
        merged_df['avg_news_sentiment'] = merged_df['avg_news_sentiment'].fillna(0.0)
        merged_df['news_sentiment_std'] = merged_df['news_sentiment_std'].fillna(0.0)
        merged_df['news_count'] = merged_df['news_count'].fillna(0).astype(int)
        
        # Fill helper columns to calculate rolling aggregates
        merged_df['total_score'] = merged_df['total_score'].fillna(0.0)
        merged_df['pos_count'] = merged_df['pos_count'].fillna(0).astype(int)
        merged_df['neg_count'] = merged_df['neg_count'].fillna(0).astype(int)
        
        # Sort chronologically before rolling window calculations
        merged_df = merged_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        
        # Group by ticker and apply rolling window of 3 trading days, then shift by 1 to prevent leakage
        grouped = merged_df.groupby('ticker')
        rolling_total_score = grouped['total_score'].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
        rolling_article_count = grouped['news_count'].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
        rolling_pos_count = grouped['pos_count'].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
        rolling_neg_count = grouped['neg_count'].rolling(window=3, min_periods=1).sum().shift(1).reset_index(level=0, drop=True)
        
        # Calculate final 3D rolling sentiment metrics
        merged_df['news_count_3d'] = rolling_article_count.fillna(0.0).astype(int)
        merged_df['avg_sentiment_3d'] = (rolling_total_score / rolling_article_count.replace(0, np.nan)).fillna(0.0)
        merged_df['pos_ratio_3d'] = (rolling_pos_count / rolling_article_count.replace(0, np.nan)).fillna(0.0)
        merged_df['neg_ratio_3d'] = (rolling_neg_count / rolling_article_count.replace(0, np.nan)).fillna(0.0)
        
        # Drop temporary aggregation columns from merged_df (keep in database only)
        merged_df = merged_df.drop(columns=['total_score', 'pos_count', 'neg_count'])
        
        # Upsert daily aggregates to sqlite table daily_aggregated_news
        try:
            from src.utils.db import upsert_daily_aggregated_news
            upsert_daily_aggregated_news(merged_df[['ticker', 'date', 'avg_news_sentiment', 'news_sentiment_std', 'news_count', 
                                                     'avg_sentiment_3d', 'news_count_3d', 'pos_ratio_3d', 'neg_ratio_3d']])
        except Exception as e:
            logger.error(f"Error upserting daily aggregated news: {e}", exc_info=True)

        # Merge transcripts (left join)
        merged_df = pd.merge(merged_df, transcript_sentiment, on=['ticker', 'date'], how='left')
        
        # Stepwise Forward Fill earnings transcript features ticker-by-ticker
        # (prevents look-ahead bias: forward fill only, no backfilling)
        transcript_cols = ['transcript_sentiment', 'transcript_length', 'positive_word_ratio', 'negative_word_ratio']
        merged_df[transcript_cols] = merged_df.groupby('ticker')[transcript_cols].ffill()
        
        # Fill remaining pre-earnings NaNs with neutral 0
        merged_df['transcript_sentiment'] = merged_df['transcript_sentiment'].fillna(0.0)
        merged_df['transcript_length'] = merged_df['transcript_length'].fillna(0.0)
        merged_df['positive_word_ratio'] = merged_df['positive_word_ratio'].fillna(0.0)
        merged_df['negative_word_ratio'] = merged_df['negative_word_ratio'].fillna(0.0)

        # Merge macro features on date
        merged_df = pd.merge(merged_df, macro_df, on='date', how='left')
        # Forward fill any weekend gaps in macro features, then backfill remaining
        macro_cols = ['macro_usdinr', 'macro_interest_rate', 'macro_cpi']
        merged_df[macro_cols] = merged_df[macro_cols].ffill().bfill()

        # 6. Drop warmup rows (first 20 rows of each ticker to clear NaNs from indicators)
        logger.info("Cleaning warmup rows (first 20 rows per ticker)...")
        
        # Sort chronologically to identify first 20 rows accurately
        merged_df = merged_df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        
        # Keep only indices where row rank within ticker is >= 20
        clean_df = merged_df.groupby('ticker').apply(lambda x: x.iloc[20:]).reset_index(drop=True)
        
        # Drop warm-down rows with NaN labels to prevent train/test contamination
        logger.info("Dropping warm-down rows where label is NaN...")
        clean_df = clean_df.dropna(subset=['label'])
        
        # 7. Set date + ticker as multi-index
        # Keep columns but index them
        final_df = clean_df.set_index(['ticker', 'date'])

        # 8. Export to Parquet
        logger.info(f"Saving final feature dataset to {self.output_parquet}...")
        final_df.to_parquet(self.output_parquet)
        logger.info("Parquet dataset written successfully.")

        # 9. Generate and save JSON report
        self._generate_report(final_df)
        
        return final_df

    def _generate_report(self, df: pd.DataFrame):
        """Generates statistical json report of features."""
        logger.info("Generating feature engineering summary report...")
        
        row_count = len(df)
        feature_cols = [col for col in df.columns if col not in ['label', 'future_return_5d']]
        feature_count = len(feature_cols)
        
        # Null percentages
        null_pct = df.isnull().mean().to_dict()
        null_pct_formatted = {k: f"{v*100:.2f}%" for k, v in null_pct.items()}
        
        # Class distribution (BUY = 1, SELL = -1, HOLD = 0)
        label_series = df['label']
        class_dist = label_series.value_counts(dropna=False).to_dict()
        
        # Convert keys to strings for JSON serializability
        class_dist_str = {}
        for k, v in class_dist.items():
            if pd.isna(k):
                class_dist_str["NaN"] = int(v)
            else:
                key_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
                class_dist_str[key_map[int(k)]] = int(v)

        report = {
            "report_generated_at": datetime.now(timezone.utc).isoformat(),
            "total_row_count": row_count,
            "feature_count": feature_count,
            "features_list": feature_cols,
            "class_distribution": class_dist_str,
            "null_percentages": null_pct_formatted
        }
        
        with open(self.report_json, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Summary report written to {self.report_json}")
