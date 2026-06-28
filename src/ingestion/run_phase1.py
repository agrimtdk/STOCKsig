import os
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import config
from src.utils.db import init_db, write_pipeline_run
from src.ingestion.price_ingestor import PriceIngestor
from src.ingestion.news_ingestor import NewsIngestor
from src.ingestion.transcript_ingestor import TranscriptIngestor

def setup_logging():
    """Configures global python logging to console and to a local log file."""
    log_file = config.log_file
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Get logger level
    level_str = config.log_level.upper()
    level = getattr(logging, level_str, logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(log_file), encoding="utf-8")
        ]
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING) # Suppress yfinance verbose logs

def main():
    setup_logging()
    logger = logging.getLogger("main_pipeline")
    
    # Generate run ID
    run_id = f"run_{uuid.uuid4().hex[:8]}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    started_at = datetime.now(timezone.utc).isoformat()
    
    logger.info(f"==================================================")
    logger.info(f"Starting Phase 1 Ingestion. Run ID: {run_id}")
    logger.info(f"==================================================")
    
    # Write initial run status
    write_pipeline_run(
        run_id=run_id,
        phase="PHASE_1_INGESTION",
        started_at=started_at,
        completed_at="",
        status="RUNNING",
        records_written=0
    )
    
    records_written = 0
    status = "SUCCESS"
    
    try:
        # Initialize Database Schema
        logger.info("Initializing database...")
        init_db()
        
        # 1. Price Ingestion
        logger.info("Running Price Ingestor...")
        price_ingestor = PriceIngestor()
        prices_count = price_ingestor.ingest_all()
        records_written += prices_count
        logger.info(f"Price Ingestor completed. Records written: {prices_count}")
        
        # 2. News Ingestion
        logger.info("Running News Ingestor...")
        news_ingestor = NewsIngestor()
        news_count = news_ingestor.ingest_all()
        records_written += news_count
        logger.info(f"News Ingestor completed. Records written: {news_count}")
        
        # 3. Transcript Ingestion
        logger.info("Running Transcript Ingestor...")
        transcript_ingestor = TranscriptIngestor()
        transcript_count = transcript_ingestor.ingest_transcripts()
        records_written += transcript_count
        logger.info(f"Transcript Ingestor completed. Records written: {transcript_count}")
        
        logger.info(f"Pipeline executed successfully. Total records written: {records_written}")
        
    except Exception as e:
        status = "FAILED"
        logger.error(f"Pipeline run {run_id} failed with error: {e}", exc_info=True)
        
    finally:
        completed_at = datetime.now(timezone.utc).isoformat()
        write_pipeline_run(
            run_id=run_id,
            phase="PHASE_1_INGESTION",
            started_at=started_at,
            completed_at=completed_at,
            status=status,
            records_written=records_written
        )
        logger.info(f"Pipeline run logged. Status: {status}, Duration: {completed_at} (UTC)")

if __name__ == "__main__":
    main()
