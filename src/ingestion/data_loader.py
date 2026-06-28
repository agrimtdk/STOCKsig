import sqlite3
import logging
import pandas as pd
from typing import Union, List, Optional
from src.utils.db import get_db_connection

logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self):
        pass

    def _format_tickers(self, tickers: Union[str, List[str]]) -> List[str]:
        """Ensures tickers parameter is formatted as a list of strings."""
        if isinstance(tickers, str):
            return [tickers]
        elif isinstance(tickers, list):
            return tickers
        else:
            raise ValueError("tickers argument must be a string or a list of strings.")

    def load_prices(
        self, 
        tickers: Union[str, List[str]], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads price data for single or multiple tickers, with optional date filtering.
        Timestamps are returned as ISO dates.
        """
        ticker_list = self._format_tickers(tickers)
        if not ticker_list:
            return pd.DataFrame()

        conn = get_db_connection()
        
        # Build SQL query dynamically to support SQL injection-safe bindings
        placeholders = ",".join("?" for _ in ticker_list)
        query = f"SELECT * FROM stock_prices WHERE ticker IN ({placeholders})"
        params = list(ticker_list)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY ticker, date ASC"

        try:
            logger.info(f"Loading prices for tickers {ticker_list} (filter: {start_date} to {end_date})...")
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Error loading price data: {e}", exc_info=True)
            raise e
        finally:
            conn.close()

    def load_news(
        self, 
        tickers: Union[str, List[str]], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads news articles for single or multiple tickers, with optional date filtering.
        """
        ticker_list = self._format_tickers(tickers)
        if not ticker_list:
            return pd.DataFrame()

        conn = get_db_connection()
        placeholders = ",".join("?" for _ in ticker_list)
        query = f"SELECT * FROM news_articles WHERE ticker IN ({placeholders})"
        params = list(ticker_list)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY ticker, date DESC"

        try:
            logger.info(f"Loading news for tickers {ticker_list} (filter: {start_date} to {end_date})...")
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Error loading news data: {e}", exc_info=True)
            raise e
        finally:
            conn.close()

    def load_transcripts(
        self, 
        tickers: Union[str, List[str]], 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Loads earnings transcripts for single or multiple tickers, with optional date filtering.
        """
        ticker_list = self._format_tickers(tickers)
        if not ticker_list:
            return pd.DataFrame()

        conn = get_db_connection()
        placeholders = ",".join("?" for _ in ticker_list)
        query = f"SELECT * FROM earnings_transcripts WHERE ticker IN ({placeholders})"
        params = list(ticker_list)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY ticker, date DESC"

        try:
            logger.info(f"Loading transcripts for tickers {ticker_list}...")
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Error loading transcript data: {e}", exc_info=True)
            raise e
        finally:
            conn.close()
            
    def load_pipeline_runs(self) -> pd.DataFrame:
        """Loads all logged pipeline runs."""
        conn = get_db_connection()
        try:
            df = pd.read_sql_query("SELECT * FROM pipeline_runs ORDER BY started_at DESC", conn)
            return df
        except Exception as e:
            logger.error(f"Error loading pipeline runs: {e}", exc_info=True)
            raise e
        finally:
            conn.close()
