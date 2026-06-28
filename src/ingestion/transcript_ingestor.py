import os
import glob
import logging
import random
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd
from src.utils.config import config
from src.utils.db import upsert_transcripts

logger = logging.getLogger(__name__)

class TranscriptIngestor:
    def __init__(self):
        self.tickers = config.tickers
        self.transcripts_dir = config.transcripts_dir
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def _generate_mock_transcript_content(self, ticker: str, year: int, quarter: str) -> str:
        """Generates a high-quality simulated earnings call transcript."""
        clean_ticker = ticker.replace(".NS", "")
        
        # Financial templates depending on ticker sector
        if "BANK" in clean_ticker or clean_ticker == "SBIN":
            sector_metrics = {
                "rev_growth": random.uniform(8.0, 16.0),
                "margin_metric": f"Net Interest Margin (NIM) stood at {random.uniform(3.2, 4.1):.2f}%",
                "operating_highlight": "strong growth in retail loans and credit card division",
                "challenge": "slightly elevated provisions due to retail asset slippages",
                "guidance": "credit growth guidance of 14-16% for the next fiscal year"
            }
        elif clean_ticker in ["TCS", "INFY"]:
            sector_metrics = {
                "rev_growth": random.uniform(5.0, 12.0),
                "margin_metric": f"Operating margins were at {random.uniform(22.0, 26.5):.1f}%",
                "operating_highlight": "large deals total contract value (TCV) hitting record highs in cloud and AI services",
                "challenge": "wage hikes and supply-side constraints continuing to pressure short-term margins",
                "guidance": "revenue growth guidance of 6-8% in constant currency terms"
            }
        else:
            # FMCG / Oil / Infra
            sector_metrics = {
                "rev_growth": random.uniform(6.0, 18.0),
                "margin_metric": f"EBITDA margins improved by {random.uniform(50, 180):.0f} bps",
                "operating_highlight": "robust volume growth in premium portfolios and capacity expansion",
                "challenge": "volatile raw material costs and inflation impacting rural demand recovery",
                "guidance": "capital expenditure plan of 15,000 crores to expand domestic capacities"
            }

        transcript = f"""=== {clean_ticker} Q{quarter} {year} Earnings Conference Call ===
Date: {year}-{3*int(quarter[-1]):02d}-15
Speakers:
- CEO: Rajesh Kumar
- CFO: Amit Sharma

--- Executive Remarks ---
Rajesh Kumar (CEO):
"Welcome everyone to our Q{quarter} {year} earnings call. I am pleased to report that {clean_ticker} has delivered another resilient quarter. Our consolidated revenue grew by {sector_metrics['rev_growth']:.1f}% year-on-year. We saw {sector_metrics['operating_highlight']}. While we faced headwind in some areas, our core business remains robust, and we are well-positioned to capitalize on structural long-term opportunities."

Amit Sharma (CFO):
"Thank you, Rajesh. Talking about financials, our revenue for the quarter was solid, and {sector_metrics['margin_metric']}. We maintained strict cost discipline. However, we did experience {sector_metrics['challenge']}. Our balance sheet remains healthy, and the board is pleased to announce an interim dividend."

--- Analyst Q&A ---
Analyst:
"Congratulations on the results. Can you share more color on the demand outlook and guidance?"

Rajesh Kumar (CEO):
"Thank you. Overall, we see healthy demand pipelines. For the upcoming periods, we are projecting {sector_metrics['guidance']}. We are confident that our digital initiatives and cost optimizations will drive sustainable shareholder value."
"""
        return transcript

    def generate_and_archive_mocks(self) -> int:
        """Generates mock transcript txt files in the raw folder for all stocks, years, and quarters."""
        logger.info("No local transcript files found. Commencing generation of mock historical transcripts...")
        current_year = datetime.now().year
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        
        # Generate for last 5 years
        years = list(range(current_year - config.start_years_ago, current_year + 1))
        
        generated_count = 0
        for ticker in self.tickers:
            for year in years:
                for q in quarters:
                    # Skip future quarters in the current year
                    if year == current_year and int(q[-1]) > (datetime.now().month - 1) // 3 + 1:
                        continue
                        
                    content = self._generate_mock_transcript_content(ticker, year, q)
                    
                    # File name matching template: {ticker}_{year}_{quarter}.txt
                    filename = f"{ticker}_{year}_{q}.txt"
                    filepath = self.transcripts_dir / filename
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                        
                    generated_count += 1
                    
        logger.info(f"Generated {generated_count} simulated transcript files at {self.transcripts_dir}")
        return generated_count

    def ingest_transcripts(self) -> int:
        """Loads transcripts from raw text files, parses them, and inserts into DB."""
        # Find files matching the wildcard {ticker}_{year}_{quarter}.txt
        # e.g. RELIANCE.NS_2025_Q1.txt
        search_pattern = str(self.transcripts_dir / "*.txt")
        files = glob.glob(search_pattern)

        # If empty, generate simulated archives first
        if not files:
            self.generate_and_archive_mocks()
            files = glob.glob(search_pattern)
            
        transcripts_list = []
        for filepath in files:
            path = Path(filepath)
            filename = path.stem # e.g. "RELIANCE.NS_2025_Q1"
            
            # Parse parts from filename
            parts = filename.split("_")
            if len(parts) >= 3:
                ticker = parts[0]
                year = int(parts[1])
                quarter = parts[2]
            else:
                # Fallback if structure is slightly different
                logger.warning(f"Could not parse ticker, year, quarter from file: {path.name}")
                continue
                
            # Date of call: set to mid-quarter month
            q_num = int(quarter[-1])
            month = 3 * q_num
            call_date = f"{year}-{month:02d}-15"
            
            # Normalize to UTC
            try:
                date_utc = pd.to_datetime(call_date).tz_localize('UTC').strftime('%Y-%m-%d')
            except Exception:
                date_utc = call_date
            
            # Read full text
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                
            transcripts_list.append({
                "id": f"{ticker}_{year}_{quarter}",
                "ticker": ticker,
                "date": date_utc,
                "quarter": quarter,
                "year": year,
                "content": content
            })

        df = pd.DataFrame(transcripts_list)
        if df.empty:
            return 0
            
        # Enforce unique id constraint
        df = df.drop_duplicates(subset=["id"])
        
        written = upsert_transcripts(df)
        return written

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = TranscriptIngestor()
    ingestor.ingest_transcripts()
