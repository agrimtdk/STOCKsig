import os
import sqlite3
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.config import config
from src.utils.db import init_db, get_db_connection, upsert_prices, upsert_news, upsert_transcripts
from src.ingestion.price_ingestor import PriceIngestor
from src.ingestion.news_ingestor import NewsIngestor
from src.ingestion.transcript_ingestor import TranscriptIngestor
from src.ingestion.data_loader import DataLoader

@pytest.fixture(autouse=True)
def setup_test_env(tmp_path, monkeypatch):
    """Overrides config paths to use a temporary sandbox directory for all tests."""
    test_db = tmp_path / "test_stock_signals.sqlite"
    test_raw = tmp_path / "raw"
    test_prices = test_raw / "prices"
    test_news = test_raw / "news"
    test_transcripts = test_raw / "transcripts"
    test_log = tmp_path / "test_phase1.log"
    
    # Ensure folders exist
    test_prices.mkdir(parents=True, exist_ok=True)
    test_news.mkdir(parents=True, exist_ok=True)
    test_transcripts.mkdir(parents=True, exist_ok=True)
    
    # Monkeypatch config properties on the Config class level
    from src.utils.config import Config
    monkeypatch.setattr(Config, "database_path", property(lambda self: test_db))
    monkeypatch.setattr(Config, "raw_dir", property(lambda self: test_raw))
    monkeypatch.setattr(Config, "prices_dir", property(lambda self: test_prices))
    monkeypatch.setattr(Config, "news_dir", property(lambda self: test_news))
    monkeypatch.setattr(Config, "transcripts_dir", property(lambda self: test_transcripts))
    monkeypatch.setattr(Config, "tickers", property(lambda self: ["TCS.NS", "INFY.NS"]))
    monkeypatch.setattr(Config, "log_file", property(lambda self: test_log))
    monkeypatch.setattr(Config, "start_years_ago", property(lambda self: 1))

def test_database_initialization():
    """Validates that all tables and indices are created correctly on initialization."""
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Verify tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    assert "stock_prices" in tables
    assert "news_articles" in tables
    assert "earnings_transcripts" in tables
    assert "pipeline_runs" in tables
    
    # Verify stock_prices schema / primary key index
    cursor.execute("PRAGMA index_list('stock_prices')")
    indices = [row[1] for row in cursor.fetchall()]
    assert any("idx_prices_ticker_date" in idx or "sqlite_autoindex" in idx for idx in indices)
    
    conn.close()

def test_price_ingestion_and_idempotency():
    """Tests downloading, UTC normalization, archiving, and idempotency of prices."""
    init_db()
    
    # Create sample mock yfinance data
    dates = pd.date_range(start="2025-01-01", end="2025-01-03", freq="D")
    mock_df = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0],
        "High": [105.0, 106.0, 107.0],
        "Low": [95.0, 96.0, 97.0],
        "Close": [102.0, 103.0, 104.0],
        "Adj Close": [102.0, 103.0, 104.0],
        "Volume": [1000, 1100, 1200]
    }, index=dates)
    mock_df.index.name = "Date"
    
    with patch("yfinance.download", return_value=mock_df) as mock_download:
        ingestor = PriceIngestor()
        written = ingestor.ingest_ticker("TCS.NS", "2025-01-01", "2025-01-03")
        
        # Verify download mock was called
        mock_download.assert_called_once()
        assert written == 3
        
        # Verify raw Parquet file was archived
        parquet_path = config.prices_dir / "TCS.NS.parquet"
        assert parquet_path.exists()
        
        # Verify db insert
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stock_prices WHERE ticker='TCS.NS'")
        assert cursor.fetchone()[0] == 3
        
        # Test idempotency (upsert logic): run ingestion again with same dates but changed close prices
        mock_df_modified = mock_df.copy()
        mock_df_modified["Close"] = [110.0, 111.0, 112.0]
        
        with patch("yfinance.download", return_value=mock_df_modified):
            written_again = ingestor.ingest_ticker("TCS.NS", "2025-01-01", "2025-01-03")
            assert written_again == 3
            
            # Check database count remains 3 (no duplication)
            cursor.execute("SELECT COUNT(*) FROM stock_prices WHERE ticker='TCS.NS'")
            assert cursor.fetchone()[0] == 3
            
            # Verify close prices were updated to modified values (upsert worked!)
            cursor.execute("SELECT close FROM stock_prices WHERE date='2025-01-01' AND ticker='TCS.NS'")
            assert cursor.fetchone()[0] == 110.0
            
        conn.close()

def test_news_ingestion_and_fallback():
    """Tests news article fetching, yfinance fallback, and mock enrichment."""
    init_db()
    
    import time
    # Prepare dummy yfinance news (yesterday's timestamp)
    mock_yf_news = [
        {
            "uuid": "news_123",
            "title": "TCS Wins Digital Transformation Deal",
            "publisher": "Reuters",
            "link": "https://reuters.com/tcs-deal",
            "providerPublishTime": int(time.time() - 86400),
            "summary": "TCS announces major deal with UK retail company."
        }
    ]
    
    with patch("yfinance.Ticker") as mock_ticker_cls:
        # Mock yf.Ticker(ticker).news
        mock_ticker_inst = MagicMock()
        mock_ticker_inst.news = mock_yf_news
        mock_ticker_cls.return_value = mock_ticker_inst
        
        ingestor = NewsIngestor()
        written = ingestor.ingest_ticker("TCS.NS")
        
        # Should write the 1 yfinance news article + 50 simulated articles due to enrichment fallback
        assert written == 51
        
        # Verify JSON archive exists
        json_path = config.news_dir / "TCS.NS.json"
        assert json_path.exists()
        
        # Verify DB entries
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM news_articles WHERE ticker='TCS.NS'")
        assert cursor.fetchone()[0] == 51
        conn.close()

def test_transcript_ingestion_and_fallback():
    """Tests earnings transcript generation, text archiving, and SQLite upsert."""
    init_db()
    
    ingestor = TranscriptIngestor()
    
    # Verify that initially there are no files in transcripts dir
    assert len(list(config.transcripts_dir.glob("*.txt"))) == 0
    
    # Triggers transcript ingestion which should generate mocks when directory is empty
    written = ingestor.ingest_transcripts()
    
    # We have 2 tickers ("TCS.NS", "INFY.NS") and start_years_ago is 1.
    # Years: current_year - 1 and current_year (e.g. 2025 and 2026).
    # Generates multiple quarters per ticker.
    assert written > 0
    
    # Verify files created in the archival layer
    text_files = list(config.transcripts_dir.glob("*.txt"))
    assert len(text_files) == written
    
    # Check that a sample file contains the right text formatting
    with open(text_files[0], "r", encoding="utf-8") as f:
        content = f.read()
        assert "Rajesh Kumar" in content
        assert "Amit Sharma" in content
        
    # Verify DB contains the written transcripts
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM earnings_transcripts")
    db_count = cursor.fetchone()[0]
    assert db_count == written
    
    # Test unique constraint (idempotency)
    written_second_time = ingestor.ingest_transcripts()
    assert written_second_time == written
    cursor.execute("SELECT COUNT(*) FROM earnings_transcripts")
    assert cursor.fetchone()[0] == db_count
    
    conn.close()

def test_dataloader_queries():
    """Tests DataLoader functionalities including multi-ticker loading and date range filtering."""
    init_db()
    
    # Seed mock data
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Seed prices
    cursor.executemany(
        "INSERT INTO stock_prices (ticker, date, open, high, low, close, adj_close, volume) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("TCS.NS", "2025-01-01", 100, 105, 95, 102, 102, 1000),
            ("TCS.NS", "2025-01-02", 102, 106, 96, 104, 104, 1100),
            ("TCS.NS", "2025-01-03", 104, 108, 98, 105, 105, 1200),
            ("INFY.NS", "2025-01-01", 50, 55, 45, 52, 52, 2000),
            ("INFY.NS", "2025-01-02", 52, 56, 46, 54, 54, 2100),
        ]
    )
    conn.commit()
    conn.close()
    
    loader = DataLoader()
    
    # 1. Single ticker query
    df_single = loader.load_prices("TCS.NS")
    assert len(df_single) == 3
    assert (df_single["ticker"] == "TCS.NS").all()
    
    # 2. Multi ticker query
    df_multi = loader.load_prices(["TCS.NS", "INFY.NS"])
    assert len(df_multi) == 5
    assert set(df_multi["ticker"].unique()) == {"TCS.NS", "INFY.NS"}
    
    # 3. Date filtering
    df_date_filter = loader.load_prices("TCS.NS", start_date="2025-01-02", end_date="2025-01-03")
    assert len(df_date_filter) == 2
    assert set(df_date_filter["date"]) == {"2025-01-02", "2025-01-03"}
    
    # 4. Out of range date filter
    df_empty = loader.load_prices("TCS.NS", start_date="2025-01-05")
    assert df_empty.empty
