import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from src.utils.config import config

logger = logging.getLogger(__name__)

def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection to the configured database file."""
    db_path = config.database_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema with necessary tables, indices, and constraints."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # 1. stock_prices table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume INTEGER,
                PRIMARY KEY (ticker, date)
            )
        """)
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_prices_ticker_date ON stock_prices (ticker, date)")

        # 2. news_articles table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id TEXT PRIMARY KEY,
                ticker TEXT,
                date TEXT,
                headline TEXT,
                summary TEXT,
                source TEXT,
                url TEXT
            )
        """)

        # 3. earnings_transcripts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS earnings_transcripts (
                id TEXT PRIMARY KEY,
                ticker TEXT,
                date TEXT,
                quarter TEXT,
                year INTEGER,
                content TEXT
            )
        """)

        # 4. pipeline_runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                phase TEXT,
                started_at TEXT,
                completed_at TEXT,
                status TEXT,
                records_written INTEGER
            )
        """)

        # 5. daily_aggregated_news table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_aggregated_news (
                ticker TEXT,
                date TEXT,
                avg_news_sentiment REAL,
                news_sentiment_std REAL,
                news_count INTEGER,
                avg_sentiment_3d REAL,
                news_count_3d INTEGER,
                pos_ratio_3d REAL,
                neg_ratio_3d REAL,
                PRIMARY KEY (ticker, date)
            )
        """)

        conn.commit()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error initializing database schema: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

def upsert_prices(prices_df) -> int:
    """Idempotently upserts historical stock prices into stock_prices."""
    if prices_df.empty:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Required columns format: ticker, date, open, high, low, close, adj_close, volume
    query = """
        INSERT INTO stock_prices (ticker, date, open, high, low, close, adj_close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            open=excluded.open,
            high=excluded.high,
            low=excluded.low,
            close=excluded.close,
            adj_close=excluded.adj_close,
            volume=excluded.volume
    """
    
    records = []
    for _, row in prices_df.iterrows():
        # Ensure timestamp is string in ISO format YYYY-MM-DD
        date_str = row['date']
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d')
        elif isinstance(date_str, str):
            date_str = date_str[:10]  # Take YYYY-MM-DD
            
        records.append((
            row['ticker'],
            date_str,
            float(row['open']),
            float(row['high']),
            float(row['low']),
            float(row['close']),
            float(row['adj_close']),
            int(row['volume'])
        ))
        
    try:
        cursor.executemany(query, records)
        conn.commit()
        logger.info(f"Upserted {len(records)} price records into stock_prices.")
        return len(records)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error upserting prices: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

def upsert_news(news_df) -> int:
    """Idempotently upserts news articles into news_articles."""
    if news_df.empty:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO news_articles (id, ticker, date, headline, summary, source, url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            ticker=excluded.ticker,
            date=excluded.date,
            headline=excluded.headline,
            summary=excluded.summary,
            source=excluded.source,
            url=excluded.url
    """
    
    records = []
    for _, row in news_df.iterrows():
        date_str = row['date']
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d')
        elif isinstance(date_str, str):
            date_str = date_str[:10]
            
        records.append((
            row['id'],
            row['ticker'],
            date_str,
            row['headline'],
            row['summary'],
            row['source'],
            row['url']
        ))
        
    try:
        cursor.executemany(query, records)
        conn.commit()
        logger.info(f"Upserted {len(records)} news records into news_articles.")
        return len(records)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error upserting news: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

def upsert_daily_aggregated_news(agg_df) -> int:
    """Idempotently upserts daily aggregated news sentiments into daily_aggregated_news."""
    if agg_df.empty:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO daily_aggregated_news (
            ticker, date, avg_news_sentiment, news_sentiment_std, news_count,
            avg_sentiment_3d, news_count_3d, pos_ratio_3d, neg_ratio_3d
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date) DO UPDATE SET
            avg_news_sentiment=excluded.avg_news_sentiment,
            news_sentiment_std=excluded.news_sentiment_std,
            news_count=excluded.news_count,
            avg_sentiment_3d=excluded.avg_sentiment_3d,
            news_count_3d=excluded.news_count_3d,
            pos_ratio_3d=excluded.pos_ratio_3d,
            neg_ratio_3d=excluded.neg_ratio_3d
    """
    
    records = []
    for _, row in agg_df.iterrows():
        date_str = row['date']
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d')
        elif isinstance(date_str, str):
            date_str = date_str[:10]
            
        records.append((
            row['ticker'],
            date_str,
            float(row.get('avg_news_sentiment', 0.0)),
            float(row.get('news_sentiment_std', 0.0)),
            int(row.get('news_count', 0)),
            float(row.get('avg_sentiment_3d', 0.0)),
            int(row.get('news_count_3d', 0)),
            float(row.get('pos_ratio_3d', 0.0)),
            float(row.get('neg_ratio_3d', 0.0))
        ))
        
    try:
        cursor.executemany(query, records)
        conn.commit()
        logger.info(f"Upserted {len(records)} daily aggregated news records into daily_aggregated_news.")
        return len(records)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error upserting daily aggregated news: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

def upsert_transcripts(transcripts_df) -> int:
    """Idempotently upserts earnings transcripts into earnings_transcripts."""
    if transcripts_df.empty:
        return 0
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = """
        INSERT INTO earnings_transcripts (id, ticker, date, quarter, year, content)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            date=excluded.date,
            quarter=excluded.quarter,
            year=excluded.year,
            content=excluded.content
    """
    
    records = []
    for _, row in transcripts_df.iterrows():
        date_str = row['date']
        if hasattr(date_str, 'strftime'):
            date_str = date_str.strftime('%Y-%m-%d')
        elif isinstance(date_str, str):
            date_str = date_str[:10]
            
        records.append((
            row['id'],
            row['ticker'],
            date_str,
            row['quarter'],
            int(row['year']),
            row['content']
        ))
        
    try:
        cursor.executemany(query, records)
        conn.commit()
        logger.info(f"Upserted {len(records)} transcripts into earnings_transcripts.")
        return len(records)
    except Exception as e:
        conn.rollback()
        logger.error(f"Error upserting transcripts: {e}", exc_info=True)
        raise e
    finally:
        conn.close()

def write_pipeline_run(run_id: str, phase: str, started_at: str, completed_at: str, status: str, records_written: int):
    """Logs a pipeline run execution state to database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO pipeline_runs (run_id, phase, started_at, completed_at, status, records_written)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id) DO UPDATE SET
            completed_at=excluded.completed_at,
            status=excluded.status,
            records_written=excluded.records_written
    """
    try:
        cursor.execute(query, (run_id, phase, started_at, completed_at, status, records_written))
        conn.commit()
        logger.debug(f"Pipeline run {run_id} logged: status={status}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error logging pipeline run: {e}", exc_info=True)
    finally:
        conn.close()
