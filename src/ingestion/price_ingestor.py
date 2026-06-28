import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.utils.config import config
from src.utils.db import upsert_prices
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)

class PriceIngestor:
    def __init__(self):
        self.tickers = config.tickers
        self.prices_dir = config.prices_dir
        self.prices_dir.mkdir(parents=True, exist_ok=True)
        
    @retry_with_backoff(exceptions=(Exception,))
    def _fetch_ticker_prices(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Downloads historical OHLCV data from yfinance for a single ticker with retries."""
        logger.info(f"Downloading historical price data for {ticker} from {start_date} to {end_date}...")
        
        # Download data
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        return df

    def ingest_ticker(self, ticker: str, start_date: str, end_date: str) -> int:
        """Ingests a single ticker, saves raw parquet, normalizes to UTC, and upserts into database."""
        try:
            df = self._fetch_ticker_prices(ticker, start_date, end_date)
            
            if df.empty:
                logger.warning(f"No price data found for {ticker} from {start_date} to {end_date}.")
                return 0
            
            # Archive raw data in Parquet
            parquet_path = self.prices_dir / f"{ticker}.parquet"
            df.to_parquet(parquet_path, index=True)
            logger.info(f"Saved raw price archive for {ticker} to {parquet_path}")

            # Normalize column names & Index
            df = df.reset_index()
            # If yfinance returned MultiIndex columns (sometimes happens with yfinance versions), flatten it
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
                
            df.columns = [col.lower().replace(" ", "_") for col in df.columns]
            
            # Rename 'date' or 'datetime' to 'date' if needed
            if 'date' not in df.columns:
                # yfinance returns 'Date' or 'Datetime' as the index, which reset_index makes a column
                if 'datetime' in df.columns:
                    df = df.rename(columns={'datetime': 'date'})
                else:
                    # Fallback for first column if named something else
                    df = df.rename(columns={df.columns[0]: 'date'})
            
            # Ensure price columns exist
            if 'adj_close' not in df.columns and 'close' in df.columns:
                df['adj_close'] = df['close']
            
            # Convert date to UTC
            # Since daily prices represent a business calendar day, we localize tz-naive dates directly to UTC
            # to prevent date-shifting (e.g. midnight in Asia/Kolkata becoming the previous day in UTC).
            dates_utc = pd.to_datetime(df['date'])
            if dates_utc.dt.tz is None:
                dates_utc = dates_utc.dt.tz_localize('UTC')
            else:
                dates_utc = dates_utc.dt.tz_convert('UTC')
            
            df['date'] = dates_utc.dt.strftime('%Y-%m-%d')
            df['ticker'] = ticker

            # Reorder/filter columns to match DB schema
            required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            df_db = df[required_cols].copy()
            
            # Drop rows with null dates or tickers
            df_db = df_db.dropna(subset=['ticker', 'date'])
            
            # Deduplicate just in case
            df_db = df_db.drop_duplicates(subset=['ticker', 'date'])
            
            # Write to DB
            written = upsert_prices(df_db)
            return written
            
        except Exception as e:
            logger.error(f"Failed to ingest price data for {ticker}: {e}", exc_info=True)
            return 0

    def ingest_all(self) -> int:
        """Ingests historical price data for all configured stocks."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=config.start_years_ago * 365)).strftime('%Y-%m-%d')
        
        total_written = 0
        for ticker in self.tickers:
            written = self.ingest_ticker(ticker, start_date, end_date)
            total_written += written
            
        logger.info(f"Price ingestion completed. Total records written: {total_written}")
        return total_written

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = PriceIngestor()
    ingestor.ingest_all()
