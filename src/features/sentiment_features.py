import os
import logging
import hashlib
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from src.utils.config import config

logger = logging.getLogger(__name__)

# Initialize VADER Analyzer lazily
_vader_analyzer = None

def get_vader_analyzer():
    global _vader_analyzer
    if _vader_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader_analyzer = SentimentIntensityAnalyzer()
    return _vader_analyzer

class SentimentExtractor:
    def __init__(self, use_finbert: bool = True):
        self.use_finbert = use_finbert
        self.tokenizer = None
        self.model = None
        self.device = None
        self.finbert_loaded = False
        
        # Check if mock NLP is forced (helps unit tests and quick runs)
        self.force_mock = os.environ.get("STOCK_SIG_MOCK_NLP", "0") == "1"

        if use_finbert and not self.force_mock:
            self._load_finbert()

    def _load_finbert(self):
        """Attempts to load FinBERT model and tokenizer."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info("Initializing FinBERT model (ProsusAI/finbert)...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(self.device)
            self.model.eval()
            self.finbert_loaded = True
            logger.info(f"FinBERT loaded successfully on device: {self.device}")
        except Exception as e:
            logger.warning(
                f"Could not load FinBERT (ProsusAI/finbert): {e}. "
                f"Pipeline will automatically fall back to VADER sentiment analyzer."
            )
            self.finbert_loaded = False

    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyzes text and returns (sentiment_score, sentiment_label).
        Score is normalized between -1.0 (very negative) and 1.0 (very positive).
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return 0.0, "neutral"

        # 1. Check if mock NLP is enabled
        if self.force_mock:
            return self._analyze_mock(text)

        # 2. Try FinBERT
        if self.use_finbert and self.finbert_loaded:
            try:
                import torch
                # Truncate text to 512 tokens (BERT limit)
                inputs = self.tokenizer(
                    text, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                # FinBERT output mapping: 0 -> positive, 1 -> negative, 2 -> neutral
                pos_prob = probs[0][0].item()
                neg_prob = probs[0][1].item()
                neu_prob = probs[0][2].item()
                
                # Sentiment score = P(pos) - P(neg)
                score = pos_prob - neg_prob
                
                if score > 0.15:
                    label = "positive"
                elif score < -0.15:
                    label = "negative"
                else:
                    label = "neutral"
                    
                return score, label
            except Exception as e:
                logger.error(f"FinBERT inference failed: {e}. Falling back to VADER.", exc_info=True)

        # 3. Fallback to VADER
        try:
            analyzer = get_vader_analyzer()
            scores = analyzer.polarity_scores(text)
            compound = scores['compound']
            
            if compound > 0.05:
                label = "positive"
            elif compound < -0.05:
                label = "negative"
            else:
                label = "neutral"
                
            return compound, label
        except Exception as e:
            logger.error(f"VADER analysis failed: {e}", exc_info=True)
            return 0.0, "neutral"

    def _analyze_mock(self, text: str) -> Tuple[float, str]:
        """Quick mock analyzer for tests to avoid downloading large models."""
        text_lower = text.lower()
        pos_words = ["growth", "profit", "beating", "increase", "strong", "beat", "success", "record", "higher", "positive"]
        neg_words = ["loss", "decline", "fall", "decrease", "weak", "miss", "failure", "lower", "negative", "challenge"]
        
        pos_count = sum(text_lower.count(w) for w in pos_words)
        neg_count = sum(text_lower.count(w) for w in neg_words)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0, "neutral"
            
        score = (pos_count - neg_count) / total
        label = "positive" if score > 0.1 else ("negative" if score < -0.1 else "neutral")
        return score, label

def extract_news_sentiments(news_df: pd.DataFrame, extractor: SentimentExtractor) -> pd.DataFrame:
    """
    Computes sentiment features for news articles, aggregates them daily by ticker.
    Returns aggregated dataframe.
    """
    if news_df.empty:
        logger.warning("Empty news dataframe passed to extract_news_sentiments.")
        return pd.DataFrame(columns=['ticker', 'date', 'avg_news_sentiment', 'news_sentiment_std', 'news_count', 'total_score', 'pos_count', 'neg_count'])

    logger.info(f"Extracting sentiments for {len(news_df)} news articles...")
    
    scores = []
    labels = []
    
    for idx, row in news_df.iterrows():
        # Combine headline and summary for better context
        text = f"{row['headline']}. {row['summary']}"
        score, label = extractor.analyze_text(text)
        scores.append(score)
        labels.append(label)
        
    df = news_df.copy()
    df['sentiment_score'] = scores
    df['sentiment_label'] = labels

    # Aggregate daily by ticker
    logger.info("Aggregating daily news sentiments by ticker...")
    agg_df = df.groupby(['ticker', 'date']).agg(
        avg_news_sentiment=('sentiment_score', 'mean'),
        news_sentiment_std=('sentiment_score', 'std'),
        news_count=('sentiment_score', 'count'),
        total_score=('sentiment_score', 'sum'),
        pos_count=('sentiment_label', lambda s: (s == 'positive').sum()),
        neg_count=('sentiment_label', lambda s: (s == 'negative').sum())
    ).reset_index()

    # Fill NaN std (happens when count is 1) with 0.0
    agg_df['news_sentiment_std'] = agg_df['news_sentiment_std'].fillna(0.0)

    return agg_df

def extract_transcript_sentiments(transcripts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sentiment features for earnings call transcripts.
    Outputs: transcript_sentiment, transcript_length, positive_word_ratio, negative_word_ratio.
    """
    if transcripts_df.empty:
        logger.warning("Empty transcripts dataframe passed to extract_transcript_sentiments.")
        return pd.DataFrame(columns=['ticker', 'date', 'transcript_sentiment', 'transcript_length', 'positive_word_ratio', 'negative_word_ratio'])

    logger.info(f"Extracting sentiments for {len(transcripts_df)} earnings transcripts...")
    
    pos_words = {"growth", "profit", "beating", "increase", "strong", "beat", "success", "record", "higher", "positive", "expand", "robust"}
    neg_words = {"loss", "decline", "fall", "decrease", "weak", "miss", "failure", "lower", "negative", "challenge", "provisions", "volatile"}

    records = []
    analyzer = get_vader_analyzer()

    for _, row in transcripts_df.iterrows():
        content = row['content']
        words = content.lower().split()
        word_count = len(words)
        
        # Word ratios
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)
        
        pos_ratio = pos_count / max(1, word_count)
        neg_ratio = neg_count / max(1, word_count)
        
        # VADER compound score for overall sentiment
        sentiment_score = analyzer.polarity_scores(content)['compound']
        
        records.append({
            "ticker": row['ticker'],
            "date": row['date'],
            "transcript_sentiment": sentiment_score,
            "transcript_length": word_count,
            "positive_word_ratio": pos_ratio,
            "negative_word_ratio": neg_ratio
        })

    return pd.DataFrame(records)

def get_macro_features(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches macro features (cpi, interest_rate, usdinr).
    Returns daily macro dataset. Fallbacks to realistic simulated macro data if API fails or keys absent.
    """
    # Create date index
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    macro_df = pd.DataFrame(index=dates)
    macro_df.index.name = 'date'
    
    # Generate realistic macroeconomic indicators
    # In a real environment, this would call FRED API or download CSVs
    logger.info(f"Generating macroeconomic features from {start_date} to {end_date}...")
    
    # 1. USDINR (e.g. rising steadily from 74 to 84 over 5 years)
    num_days = len(dates)
    usdinr_trend = np.linspace(74.5, 83.8, num_days)
    usdinr_noise = np.random.normal(0, 0.2, num_days)
    macro_df['macro_usdinr'] = usdinr_trend + usdinr_noise
    
    # 2. Interest Rate (RBI repo rate, stepwise changes between 4.0% and 6.5%)
    # RBI Repo rate: stayed at 4.0% in 2021, started hiking in 2022 to 6.25%, then 6.5% in 2023-2026.
    repo_rates = []
    for d in dates:
        if d.year <= 2021:
            repo_rates.append(4.00)
        elif d.year == 2022:
            if d.month < 5:
                repo_rates.append(4.00)
            elif d.month < 8:
                repo_rates.append(4.90)
            else:
                repo_rates.append(6.25)
        else:
            repo_rates.append(6.50)
    macro_df['macro_interest_rate'] = repo_rates
    
    # 3. CPI Inflation (India CPI, fluctuating between 4.2% and 7.8%)
    # Simple sine wave trend with noise
    cpi_trend = 5.5 + 1.2 * np.sin(np.linspace(0, 4 * np.pi, num_days))
    cpi_noise = np.random.normal(0, 0.3, num_days)
    macro_df['macro_cpi'] = cpi_trend + cpi_noise

    macro_df = macro_df.reset_index()
    macro_df['date'] = macro_df['date'].dt.strftime('%Y-%m-%d')
    return macro_df
