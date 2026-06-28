import os
import json
import logging
import time
import yfinance as yf
import streamlit as st

logger = logging.getLogger(__name__)

FAVORITES_DIR = r"c:\Users\Agrim Sharma\Desktop\StockSig\data\user"
FAVORITES_PATH = os.path.join(FAVORITES_DIR, "favorites.json")

@st.cache_data(ttl=60)
def get_live_quote(ticker: str) -> dict:
    """
    Fetches real-time price info from yfinance.
    Attempts to use ticker.info / fast_info, falling back to history if needed.
    """
    try:
        t = yf.Ticker(ticker)
        # Try getting fast_info first (extremely fast, doesn't make blocking web requests for full profile)
        fast = getattr(t, "fast_info", None)
        
        current_price = None
        prev_close = None
        open_val = None
        high_val = None
        low_val = None
        volume_val = None
        market_cap = None
        
        if fast is not None:
            try:
                current_price = fast.get("last_price", None)
                prev_close = fast.get("previous_close", None)
                open_val = fast.get("open", None)
                high_val = fast.get("day_high", None)
                low_val = fast.get("day_low", None)
                volume_val = fast.get("last_volume", None)
                market_cap = fast.get("market_cap", None)
            except Exception as fe:
                logger.warning(f"Error accessing fast_info for {ticker}: {fe}")

        # Fallback to history if fast_info properties are missing/None
        if current_price is None or prev_close is None:
            hist = t.history(period="5d")
            if not hist.empty:
                latest_row = hist.iloc[-1]
                current_price = latest_row["Close"]
                open_val = latest_row["Open"]
                high_val = latest_row["High"]
                low_val = latest_row["Low"]
                volume_val = latest_row["Volume"]
                if len(hist) > 1:
                    prev_close = hist.iloc[-2]["Close"]
                else:
                    prev_close = open_val

        # Final sanity fallbacks if still None
        if current_price is None:
            current_price = 0.0
        if prev_close is None:
            prev_close = current_price
            
        day_change = current_price - prev_close
        day_change_pct = (day_change / prev_close * 100.0) if prev_close != 0 else 0.0
        
        # Clean numeric properties
        return {
            "current_price": float(current_price),
            "previous_close": float(prev_close),
            "open": float(open_val) if open_val is not None else float(current_price),
            "high": float(high_val) if high_val is not None else float(current_price),
            "low": float(low_val) if low_val is not None else float(current_price),
            "volume": int(volume_val) if volume_val is not None else 0,
            "day_change": float(day_change),
            "day_change_pct": float(day_change_pct),
            "market_cap": float(market_cap) if market_cap is not None else None,
            "last_updated": time.strftime("%H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error fetching live quote for {ticker}: {e}")
        return {
            "current_price": 0.0,
            "previous_close": 0.0,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "volume": 0,
            "day_change": 0.0,
            "day_change_pct": 0.0,
            "market_cap": None,
            "last_updated": "N/A"
        }

def load_favorites() -> list:
    """Loads watchlist favorite symbols list from json."""
    if not os.path.exists(FAVORITES_PATH):
        return []
    try:
        with open(FAVORITES_PATH, "r") as f:
            data = json.load(f)
            return data.get("favorites", [])
    except Exception as e:
        logger.error(f"Error loading favorites: {e}")
        return []

def save_favorites(favorites_list: list):
    """Saves watchlist favorite symbols list to json."""
    try:
        os.makedirs(FAVORITES_DIR, exist_ok=True)
        with open(FAVORITES_PATH, "w") as f:
            json.dump({"favorites": favorites_list}, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving favorites: {e}")

def toggle_favorite(ticker: str):
    """Adds ticker if missing, else removes it from favorites."""
    favs = load_favorites()
    if ticker in favs:
        favs.remove(ticker)
    else:
        favs.append(ticker)
    save_favorites(favs)
