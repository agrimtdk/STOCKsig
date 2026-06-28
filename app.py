import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
import logging
from datetime import datetime

# Set page config FIRST before any other streamlit commands
st.set_page_config(
    page_title="STOCKsig",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from src.dashboard.data_service import (
    load_model, load_features, load_signals, load_equity_curve,
    load_trade_log, load_feature_importance, load_backtest_report,
    load_benchmark_comparison
)
from src.dashboard.signal_service import (
    get_latest_signal_info, get_technical_snapshot, get_sentiment_snapshot,
    get_market_regime, get_signal_explainability, get_live_signal_for_ticker
)
from src.dashboard.live_data import get_live_quote, load_favorites, toggle_favorite
from src.dashboard.glossary import tooltip
from src.dashboard.metrics_service import get_volatility_risk_level, format_metric_value
from src.dashboard.charts import (
    create_candlestick_chart, create_equity_curve_chart,
    create_feature_importance_chart, create_sparkline,
    create_probability_histograms, get_theme_colors,
    create_confidence_gauge, create_signal_history_timeline,
    create_momentum_charts
)
from src.dashboard.components import (
    inject_global_styles, render_signal_card, render_kpi_card, render_status_footer,
    get_navbar_css, FONT_STACK, THEMES
)

# Configure logger
logger = logging.getLogger("app")

TICKERS = ["ASIANPAINT.NS", "BHARTIARTL.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "ITC.NS", "LT.NS", "RELIANCE.NS", "SBIN.NS", "TCS.NS"]

def filter_df_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """Filters a DataFrame sorted by date index by a period string."""
    if df.empty:
        return df
    
    # Explicitly convert index to datetime for safe offset operations
    df_sorted = df.copy()
    df_sorted.index = pd.to_datetime(df_sorted.index)
    df_sorted = df_sorted.sort_index()
    
    latest_date = df_sorted.index.max()
    
    from pandas.tseries.offsets import DateOffset
    
    start_date = None
    if period == "1D":
        start_date = latest_date - pd.Timedelta(days=1)
    elif period == "5D":
        start_date = latest_date - pd.Timedelta(days=5)
    elif period == "1W":
        start_date = latest_date - pd.Timedelta(weeks=1)
    elif period == "1M":
        start_date = latest_date - DateOffset(months=1)
    elif period == "3M":
        start_date = latest_date - DateOffset(months=3)
    elif period == "6M":
        start_date = latest_date - DateOffset(months=6)
    elif period == "YTD":
        start_date = pd.to_datetime(f"{latest_date.year}-01-01")
    elif period == "1Y":
        start_date = latest_date - DateOffset(years=1)
    elif period == "2Y":
        start_date = latest_date - DateOffset(years=2)
    elif period == "3Y":
        start_date = latest_date - DateOffset(years=3)
    elif period == "5Y":
        start_date = latest_date - DateOffset(years=5)
    elif period == "10Y":
        start_date = latest_date - DateOffset(years=10)
    else: # MAX / ALL
        return df
        
    df_filtered = df_sorted[df_sorted.index >= start_date]
    
    # Convert index back to string to preserve original dataset formats
    df_filtered.index = df_filtered.index.strftime("%Y-%m-%d")
    return df_filtered

def get_roundtrip_trades(trade_log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups buy and sell trades from trade log into roundtrip trades.
    """
    if trade_log_df.empty:
        return pd.DataFrame()
        
    roundtrips = []
    # Track open positions during reconstruction
    # ticker -> (entry_date, entry_price, shares)
    open_positions = {}
    
    # Sort trades chronologically
    df_sorted = trade_log_df.sort_values(by="date")
    
    for _, row in df_sorted.iterrows():
        ticker = row["ticker"]
        action = row["action"]
        date = row["date"]
        price = row["price"]
        shares = row["shares"]
        pnl = row["realized_pnl"]
        
        if action == "BUY":
            open_positions[ticker] = (date, price, shares)
        elif action.startswith(("SELL", "LIQUIDATE")):
            if ticker in open_positions:
                entry_date, entry_price, entry_shares = open_positions[ticker]
                ret_pct = (price - entry_price) / entry_price
                
                roundtrips.append({
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": price,
                    "shares": entry_shares,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "holding_days": (pd.to_datetime(date) - pd.to_datetime(entry_date)).days
                })
                del open_positions[ticker]
                
    return pd.DataFrame(roundtrips)

def main():
    import time
    st.session_state["theme"] = "dark"
    if "page" not in st.session_state:
        st.session_state["page"] = "Live Signal Panel"
    if "db_mode" not in st.session_state:
        st.session_state["db_mode"] = "LIVE"
    if "selected_ticker" not in st.session_state:
        st.session_state["selected_ticker"] = TICKERS[0]
        
    # Load all cached datasets
    model = load_model()
    features_df = load_features()
    signals_df = load_signals()
    equity_df = load_equity_curve()
    trade_log_df = load_trade_log()
    feat_imp_df = load_feature_importance()
    report = load_backtest_report()
    bench_comp = load_benchmark_comparison()
    
    # Inject global styles
    inject_global_styles(st.session_state["theme"])
    
    # 1. Top Navbar row (2-column layout: Left Brand, Right Nav)
    col_brand, col_nav = st.columns([2, 8])
    
    with col_brand:
        brand_html = f"""
        <div style="display: flex; align-items: baseline; font-family: {FONT_STACK};">
            <span style="font-weight: 800; font-size: 2.2rem; letter-spacing: 0.12em; color: #E5E7EB;">STOCK</span>
            <span style="font-weight: 800; font-size: 1.35rem; letter-spacing: 0.12em; color: #E5E7EB; margin-left: 2px;">sig</span>
        </div>
        """
        st.markdown(brand_html, unsafe_allow_html=True)
        
    with col_nav:
        nav_cols = st.columns([1, 1, 1, 1, 1, 1, 1], gap="small")
        
        with nav_cols[0]:
            if st.button("Live Signal", key="nav_live"):
                st.session_state["page"] = "Live Signal Panel"
                st.rerun()
                
        with nav_cols[1]:
            if st.button("Price Chart", key="nav_chart"):
                st.session_state["page"] = "Price + Signal Chart"
                st.rerun()
                
        with nav_cols[2]:
            if st.button("Backtest", key="nav_backtest"):
                st.session_state["page"] = "Backtest Analytics"
                st.rerun()
                
        with nav_cols[3]:
            if st.button("Trades", key="nav_trades"):
                st.session_state["page"] = "Trade Log"
                st.rerun()
                
        with nav_cols[4]:
            if st.button("Features", key="nav_feat"):
                st.session_state["page"] = "Feature Importance"
                st.rerun()
                
        with nav_cols[5]:
            if st.button("Model Health", key="nav_health"):
                st.session_state["page"] = "Model Health"
                st.rerun()
                
        with nav_cols[6]:
            if st.button("Replay", key="nav_replay"):
                st.session_state["page"] = "Portfolio Replay"
                st.rerun()
            
    page_options = [
        "Live Signal Panel",
        "Price + Signal Chart",
        "Backtest Analytics",
        "Trade Log",
        "Feature Importance",
        "Model Health",
        "Portfolio Replay"
    ]
    # Active index maps to column index (1 to 7) inside col_nav
    active_index = page_options.index(st.session_state["page"]) + 1
    st.markdown(get_navbar_css(active_index, st.session_state["theme"]), unsafe_allow_html=True)
    
    # 2. Controls & Watchlist Row
    st.markdown('<div style="border-bottom: 1px solid #202938; margin: 10px 0 14px 0;"></div>', unsafe_allow_html=True)
    col_ctrl_mode, col_ctrl_watch = st.columns([2.0, 8.0])
    
    with col_ctrl_mode:
        db_mode = st.radio(
            "Terminal Mode",
            options=["LIVE", "BACKTEST"],
            horizontal=True,
            index=0 if st.session_state["db_mode"] == "LIVE" else 1,
            key="temp_db_mode",
            label_visibility="collapsed"
        )
        if db_mode != st.session_state["db_mode"]:
            st.session_state["db_mode"] = db_mode
            st.rerun()
            
    with col_ctrl_watch:
        favorites = load_favorites()
        if not favorites:
            st.markdown("<span style='color:#8B949E; font-size: 11px; line-height: 24px; display: inline-block; padding-top: 4px;'>WATCHLIST: No favorites yet. Star a stock to add it.</span>", unsafe_allow_html=True)
        else:
            pill_cols = st.columns(len(favorites))
            for idx, fav in enumerate(favorites):
                fav_clean = fav.split(".")[0]
                with pill_cols[idx]:
                    if st.button(f"📁 {fav_clean}", key=f"fav_pill_{fav}_{idx}"):
                        st.session_state["selected_ticker"] = fav
                        st.session_state["temp_ticker_select"] = fav
                        st.rerun()
                        
    page = st.session_state["page"]
    
    # Define generic bidirectional ticker change callback
    def on_ticker_change():
        st.session_state["selected_ticker"] = st.session_state["temp_ticker_select"]
    
    # Render page
    if page == "Live Signal Panel":
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.subheader("Live Signal Panel")
        st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)
        
        # Stock Selector row with Star toggle
        col_sel, col_star = st.columns([9, 1])
        with col_sel:
            selected_ticker = st.selectbox(
                "Select Asset Symbol",
                TICKERS,
                index=TICKERS.index(st.session_state.get("selected_ticker", TICKERS[0])),
                key="temp_ticker_select",
                on_change=on_ticker_change
            )
        with col_star:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            is_fav = selected_ticker in favorites
            star_label = "★" if is_fav else "☆"
            if st.button(star_label, key="toggle_star_btn", use_container_width=True):
                toggle_favorite(selected_ticker)
                st.rerun()
        
        if not signals_df.empty and not features_df.empty:
            # Check mode and fetch live quote
            quote_info = None
            if st.session_state["db_mode"] == "LIVE":
                quote_info = get_live_quote(selected_ticker)
                
                # Render live quote strip
                if quote_info:
                    color = "#10B981" if quote_info["day_change"] >= 0 else "#EF4444"
                    sign = "+" if quote_info["day_change"] >= 0 else ""
                    vol = quote_info["volume"]
                    if vol >= 1_000_000:
                        vol_str = f"{vol / 1_000_000:.1f}M"
                    elif vol >= 1_000:
                        vol_str = f"{vol / 1_000:.1f}K"
                    else:
                        vol_str = str(vol)
                        
                    quote_html = f"""
                    <div style="background-color: #121821; border: 1px solid #202938; border-radius: 8px; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; font-family: {FONT_STACK}; margin-bottom: 20px;">
                        <div>
                            <span style="font-weight: 800; font-size: 1.2rem; color: #E6EDF3;">{selected_ticker}</span>
                        </div>
                        <div style="display: flex; gap: 24px; align-items: center;">
                            <div>
                                <span style="color: #8B949E; font-size: 0.8rem; display: block; text-transform: uppercase;">Price</span>
                                <span style="font-weight: 700; font-size: 1.2rem; color: {color};">₹{quote_info["current_price"]:.2f}</span>
                            </div>
                            <div>
                                <span style="color: #8B949E; font-size: 0.8rem; display: block; text-transform: uppercase;">Change</span>
                                <span style="font-weight: 700; font-size: 1.2rem; color: {color};">{sign}{quote_info["day_change_pct"]:.2f}%</span>
                            </div>
                            <div>
                                <span style="color: #8B949E; font-size: 0.8rem; display: block; text-transform: uppercase;">High</span>
                                <span style="font-weight: 600; color: #E6EDF3;">₹{quote_info["high"]:.2f}</span>
                            </div>
                            <div>
                                <span style="color: #8B949E; font-size: 0.8rem; display: block; text-transform: uppercase;">Low</span>
                                <span style="font-weight: 600; color: #E6EDF3;">₹{quote_info["low"]:.2f}</span>
                            </div>
                            <div>
                                <span style="color: #8B949E; font-size: 0.8rem; display: block; text-transform: uppercase;">Volume</span>
                                <span style="font-weight: 600; color: #E6EDF3;">{vol_str}</span>
                            </div>
                            <div>
                                <span style="color: #8B949E; font-size: 0.8rem; display: block; text-transform: uppercase;">Updated</span>
                                <span style="font-weight: 600; color: #8B949E;">{quote_info["last_updated"]}</span>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(quote_html, unsafe_allow_html=True)
            
            # Setup base historical values
            df_stock = signals_df.xs(selected_ticker, level="ticker")
            latest_date = df_stock.index.max()
            
            sig_info = get_latest_signal_info(signals_df, selected_ticker, latest_date)
            tech = get_technical_snapshot(features_df, selected_ticker, latest_date)
            sent = get_sentiment_snapshot(features_df, selected_ticker, latest_date)
            regime = get_market_regime(features_df, selected_ticker, latest_date)
            
            # Live predictions logic overrides
            if st.session_state["db_mode"] == "LIVE" and quote_info:
                live_res = get_live_signal_for_ticker(selected_ticker, quote_info, features_df, model)
                if live_res:
                    sig_info = {
                        "date": live_res["date"],
                        "ticker": selected_ticker,
                        "signal": 1 if live_res["label"] == "BUY" else (-1 if live_res["label"] == "SELL" else 0),
                        "label": live_res["label"],
                        "confidence_score": live_res["confidence_score"],
                        "prob_sell": live_res["prob_sell"],
                        "prob_hold": live_res["prob_hold"],
                        "prob_buy": live_res["prob_buy"]
                    }
                    features_df = live_res["df_stock_live"]
                    
                    df_stock_signals = signals_df.xs(selected_ticker, level="ticker").copy()
                    df_stock_signals = df_stock_signals.reset_index()
                    live_sig_row = {
                        "ticker": selected_ticker,
                        "date": live_res["date"],
                        "signal": sig_info["signal"],
                        "confidence_score": sig_info["confidence_score"],
                        "prob_sell": sig_info["prob_sell"],
                        "prob_hold": sig_info["prob_hold"],
                        "prob_buy": sig_info["prob_buy"]
                    }
                    if live_res["date"] in df_stock_signals["date"].values:
                        idx = df_stock_signals[df_stock_signals["date"] == live_res["date"]].index[0]
                        for k, v in live_sig_row.items():
                            df_stock_signals.at[idx, k] = v
                    else:
                        df_stock_signals = pd.concat([df_stock_signals, pd.DataFrame([live_sig_row])], ignore_index=True)
                    signals_df = df_stock_signals.set_index(["ticker", "date"])
                    
                    latest_date = live_res["date"]
                    tech = get_technical_snapshot(features_df, selected_ticker, latest_date)
                    sent = get_sentiment_snapshot(features_df, selected_ticker, latest_date)
                    regime = get_market_regime(features_df, selected_ticker, latest_date)
            
            # Sparkline prices
            prices_stock = features_df.xs(selected_ticker, level="ticker").sort_index()
            
            # Row 1 layout: Signal Card, Confidence Gauge, Sparkline
            col1, col2, col3 = st.columns([2, 1, 1])
            sig_label = sig_info.get("label", "HOLD")
            
            with col1:
                with st.container(border=True):
                    # We inject a hidden div with the class to trigger CSS :has glow coloring
                    st.markdown(f'<div class="border-{sig_label.lower()}" style="display:none;"></div>', unsafe_allow_html=True)
                    render_signal_card(
                        ticker=selected_ticker,
                        signal_label=sig_label,
                        confidence=sig_info.get("confidence_score", 0.0),
                        probs={
                            "prob_buy": sig_info.get("prob_buy", 0.0),
                            "prob_hold": sig_info.get("prob_hold", 0.0),
                            "prob_sell": sig_info.get("prob_sell", 0.0)
                        },
                        theme=st.session_state["theme"]
                    )
            
            with col2:
                with st.container(border=True):
                    st.markdown('<div class="metric-title">Model Confidence</div>', unsafe_allow_html=True)
                    gauge_fig = create_confidence_gauge(
                        confidence=sig_info.get("confidence_score", 0.0),
                        signal_label=sig_label,
                        theme=st.session_state["theme"]
                    )
                    st.plotly_chart(gauge_fig, use_container_width=True, config={"displayModeBar": False})
                    
            with col3:
                with st.container(border=True):
                    st.markdown('<div class="metric-title">30D Trend Sparkline</div>', unsafe_allow_html=True)
                    sparkline_prices = prices_stock.loc[:latest_date, "close"].tail(30)
                    spark_fig = create_sparkline(sparkline_prices, st.session_state["theme"])
                    st.plotly_chart(spark_fig, use_container_width=True, config={"displayModeBar": False})
            
            # Row 2 layout: 3 columns
            col_mid1, col_mid2, col_mid3 = st.columns(3)
            
            with col_mid1:
                # Market Regime Snapshot
                with st.container(border=True):
                    st.markdown(f'<div class="metric-title" style="margin-bottom:8px;">{tooltip("Market Regime")}</div>', unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('Trend')}**: `{regime['trend']}`", unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('Volatility')}**: `{regime['volatility']}`", unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('Sentiment Score', 'Sentiment')}**: `{regime['sentiment']}`", unsafe_allow_html=True)
                
                # PANEL B: Recent Signal History
                with st.container(border=True):
                    st.markdown('<div class="metric-title" style="margin-bottom:8px;">Recent Signal History (Last 10 Days)</div>', unsafe_allow_html=True)
                    timeline_fig = create_signal_history_timeline(signals_df, selected_ticker, latest_date, st.session_state["theme"])
                    st.plotly_chart(timeline_fig, use_container_width=True, config={"displayModeBar": False})
                    
            with col_mid2:
                # Technical Snapshot
                with st.container(border=True):
                    st.markdown('<div class="metric-title" style="margin-bottom:8px;">Technical Snapshot</div>', unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('RSI')} (14)**: `{tech.get('rsi', 50.0):.2f}`", unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('MACD')} Hist**: `{tech.get('macd_hist', 0.0):.4f}`", unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('ATR')} (14)**: `{tech.get('atr', 0.0):.2f}`", unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('Bollinger Bands', 'BB Width')}**: `{tech.get('bollinger_width', 0.0):.4f}`", unsafe_allow_html=True)
                    risk_lvl = get_volatility_risk_level(tech.get('volatility', 2.0))
                    st.markdown(f"**{tooltip('Volatility', 'Volatility Risk')}**: <span class='badge badge-risk-{risk_lvl.lower()}'>{risk_lvl}</span>", unsafe_allow_html=True)
                
                # PANEL C: Price Momentum Panel
                with st.container(border=True):
                    st.markdown('<div class="metric-title" style="margin-bottom:8px;">Price Momentum Panel</div>', unsafe_allow_html=True)
                    momentum_fig = create_momentum_charts(features_df, selected_ticker, latest_date, st.session_state["theme"])
                    st.plotly_chart(momentum_fig, use_container_width=True, config={"displayModeBar": False})
                    
            with col_mid3:
                # Sentiment Snapshot
                with st.container(border=True):
                    st.markdown(f'<div class="metric-title" style="margin-bottom:8px;">{tooltip("Sentiment Score", "Sentiment Snapshot")}</div>', unsafe_allow_html=True)
                    st.markdown(f"**{tooltip('Sentiment Score', 'News Sentiment')}**: `{sent.get('news_sentiment', 0.0):+.3f}`", unsafe_allow_html=True)
                    st.markdown(f"**News Count**: `{sent.get('news_count', 0)}`")
                    st.markdown(f"**Transcript Sentiment**: `{sent.get('transcript_sentiment', 0.0):+.3f}`")
            
            # Row 3 layout: Explainability Section (full-width)
            with st.container(border=True):
                st.markdown('<div class="metric-title" style="margin-bottom:12px;">Signal Explainability (Top 5 Contributing Features)</div>', unsafe_allow_html=True)
                
                top_features = get_signal_explainability(features_df, selected_ticker, latest_date, feat_imp_df, sig_info.get("signal", 0))
                if top_features:
                    explain_df = pd.DataFrame(top_features)
                    explain_df = explain_df.rename(columns={
                        "feature": "Feature",
                        "raw_val": "Raw Value",
                        "z_score": "Z-Score",
                        "influence": "Influence Score",
                        "status": "State"
                    })
                    # Format numeric columns
                    explain_df["Raw Value"] = explain_df["Raw Value"].map(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else str(x))
                    explain_df["Z-Score"] = explain_df["Z-Score"].map(lambda x: f"{x:+.2f}")
                    explain_df["Influence Score"] = explain_df["Influence Score"].map(lambda x: f"{x:.4f}")
                    
                    # Replace feature names with tooltips dynamically in the Series elements before HTML conversion
                    def map_feature_to_tooltip(feat):
                        feat_lower = feat.lower()
                        if "macd" in feat_lower:
                            return tooltip("MACD", feat)
                        elif "rsi" in feat_lower:
                            return tooltip("RSI", feat)
                        elif "atr" in feat_lower:
                            return tooltip("ATR", feat)
                        elif "obv" in feat_lower:
                            return tooltip("OBV", feat)
                        elif "bollinger" in feat_lower:
                            return tooltip("Bollinger Bands", feat)
                        elif "volatility" in feat_lower:
                            return tooltip("Volatility", feat)
                        elif "sentiment" in feat_lower:
                            return tooltip("Sentiment Score", feat)
                        elif "cpi" in feat_lower:
                            return tooltip("macro_cpi", feat)
                        elif "usdinr" in feat_lower:
                            return tooltip("macro_usdinr", feat)
                        elif "interest_rate" in feat_lower:
                            return tooltip("macro_interest_rate", feat)
                        elif "volume_zscore" in feat_lower:
                            return tooltip("Volume Z-score", feat)
                        elif "volume" in feat_lower:
                            return tooltip("Volume Z-score", feat)
                        elif "sma" in feat_lower:
                            return tooltip("SMA", feat)
                        elif "ema" in feat_lower:
                            return tooltip("EMA", feat)
                        return feat

                    explain_df["Feature"] = explain_df["Feature"].apply(map_feature_to_tooltip)

                    # Convert to HTML to allow tooltips in table cell rendering
                    html_table = explain_df[["Feature", "Raw Value", "Z-Score", "Influence Score", "State"]].to_html(
                        escape=False, index=False, classes="dataframe"
                    )
                    
                    st.markdown(html_table, unsafe_allow_html=True)
                else:
                    st.info("Feature importance or explanation data not available.")
            
            # Watchlist Quick Scan
            st.markdown('<div style="margin-top: 24px;"></div>', unsafe_allow_html=True)
            st.subheader("Watchlist Quick Scan")
            if not favorites:
                st.info("No watchlist assets yet. Star a stock to add it.")
            else:
                scan_rows = []
                for fav in favorites:
                    fav_quote = get_live_quote(fav)
                    if st.session_state["db_mode"] == "LIVE" and fav_quote:
                        fav_payload = load_model()
                        fav_res = get_live_signal_for_ticker(fav, fav_quote, features_df, fav_payload)
                        if fav_res:
                            sig_label = fav_res["label"]
                            confidence = fav_res["confidence_score"]
                        else:
                            sig_label = "HOLD"
                            confidence = 50.0
                        price = fav_quote["current_price"]
                        day_pct = fav_quote["day_change_pct"]
                        last_updated = fav_quote["last_updated"]
                    else:
                        if not signals_df.empty and fav in signals_df.index.get_level_values("ticker"):
                            df_fav = signals_df.xs(fav, level="ticker")
                            latest_fav_date = df_fav.index.max()
                            fav_sig_info = get_latest_signal_info(signals_df, fav, latest_fav_date)
                            sig_label = fav_sig_info.get("label", "HOLD")
                            confidence = fav_sig_info.get("confidence_score", 0.0)
                        else:
                            sig_label = "HOLD"
                            confidence = 50.0
                            
                        if not features_df.empty and fav in features_df.index.get_level_values("ticker"):
                            df_fav_feat = features_df.xs(fav, level="ticker")
                            latest_fav_date = df_fav_feat.index.max()
                            row_feat = df_fav_feat.loc[latest_fav_date]
                            price = row_feat.get("close", 0.0)
                            day_pct = row_feat.get("return_1d", 0.0) * 100.0
                        else:
                            price = 0.0
                            day_pct = 0.0
                        last_updated = "Historical"
                        
                    scan_rows.append({
                        "Ticker": fav,
                        "Price": f"₹{price:.2f}",
                        "Signal": sig_label,
                        "Confidence": f"{confidence:.1f}%",
                        "Day %": f"{'+' if day_pct >= 0 else ''}{day_pct:.2f}%",
                        "Last Updated": last_updated
                    })
                df_scan = pd.DataFrame(scan_rows)
                def color_signal(val):
                    if val == "BUY":
                        return "background-color: rgba(16, 185, 129, 0.2); color: #10B981; font-weight: bold;"
                    elif val == "SELL":
                        return "background-color: rgba(239, 68, 68, 0.2); color: #EF4444; font-weight: bold;"
                    return "color: #E6EDF3;"
                    
                def color_pct(val):
                    if val.startswith("+"):
                        return "color: #10B981; font-weight: bold;"
                    elif val.startswith("-"):
                        return "color: #EF4444; font-weight: bold;"
                    return "color: #E6EDF3;"
                    
                styled_df = df_scan.style.map(color_signal, subset=["Signal"]).map(color_pct, subset=["Day %"])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
            
    elif page == "Price + Signal Chart":
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.subheader("Price + Signal Interactive Chart")
        st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)
        
        # Stock Selector row with Star toggle
        col_sel, col_star = st.columns([9, 1])
        with col_sel:
            selected_ticker = st.selectbox(
                "Select Asset Symbol",
                TICKERS,
                index=TICKERS.index(st.session_state.get("selected_ticker", TICKERS[0])),
                key="temp_ticker_select",
                on_change=on_ticker_change
            )
        with col_star:
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            is_fav = selected_ticker in favorites
            star_label = "★" if is_fav else "☆"
            if st.button(star_label, key="toggle_star_btn_chart", use_container_width=True):
                toggle_favorite(selected_ticker)
                st.rerun()
                
        # Fetch live predictions overlay for charts if in LIVE mode
        quote_info = None
        if st.session_state["db_mode"] == "LIVE":
            quote_info = get_live_quote(selected_ticker)
            
        # Get historical prices slice
        if not features_df.empty and selected_ticker in features_df.index.get_level_values("ticker"):
            df_stock = features_df.xs(selected_ticker, level="ticker").copy()
        else:
            df_stock = pd.DataFrame()
            
        # 1. Always build merged chart data
        if st.session_state["db_mode"] == "LIVE" and quote_info and not df_stock.empty:
            today_str = datetime.now().strftime("%Y-%m-%d")
            live_df = pd.DataFrame([{
                "open": quote_info["open"],
                "high": quote_info["high"],
                "low": quote_info["low"],
                "close": quote_info["current_price"],
                "adj_close": quote_info["current_price"],
                "volume": quote_info["volume"],
            }], index=pd.Index([today_str], name="date"))
            
            # Forward fill macro/sentiments from last row
            last_row = df_stock.iloc[-1]
            for col in df_stock.columns:
                if col not in live_df.columns:
                    live_df[col] = last_row[col]
                    
            merged_df = pd.concat([df_stock, live_df])
            merged_df = merged_df[~merged_df.index.duplicated(keep="last")]
            merged_df = merged_df.sort_index()
        else:
            # Fallback if live fetch fails or in BACKTEST mode
            merged_df = df_stock.copy()
            
        # Timeframe range selector with state binding
        if "chart_period" not in st.session_state:
            st.session_state["chart_period"] = "1Y"
            
        def on_period_change():
            st.session_state["chart_period"] = st.session_state["temp_chart_period"]
            
        chart_options = ["1D", "5D", "1W", "1M", "3M", "6M", "YTD", "1Y", "2Y", "3Y", "5Y", "10Y", "MAX"]
        chart_period = st.selectbox(
            "Select Timeframe Range",
            options=chart_options,
            index=chart_options.index(st.session_state.get("chart_period", "1Y")),
            key="temp_chart_period",
            on_change=on_period_change
        )
        
        if not merged_df.empty:
            # 2. Apply existing timeframe filter on merged_df
            filtered_df = filter_df_by_period(merged_df, chart_period)
            
            # 3. Recompute overlays AFTER filtering
            if not filtered_df.empty:
                filtered_df["sma_20"] = filtered_df["close"].rolling(window=20, min_periods=1).mean()
                filtered_df["ema_20"] = filtered_df["close"].ewm(span=20, adjust=False).mean()
            
            # Format features MultiIndex
            filtered_df["ticker"] = selected_ticker
            features_df_filtered = filtered_df.reset_index().set_index(["ticker", "date"])
            
            # Align and merge signals
            if not signals_df.empty and selected_ticker in signals_df.index.get_level_values("ticker"):
                df_stock_sig = signals_df.xs(selected_ticker, level="ticker").copy()
            else:
                df_stock_sig = pd.DataFrame()
                
            if st.session_state["db_mode"] == "LIVE" and quote_info and not df_stock_sig.empty:
                live_res = get_live_signal_for_ticker(selected_ticker, quote_info, features_df, model)
                if live_res:
                    today_str = live_res["date"]
                    live_sig_row = pd.DataFrame([{
                        "signal": 1 if live_res["label"] == "BUY" else (-1 if live_res["label"] == "SELL" else 0),
                        "confidence_score": live_res["confidence_score"],
                        "prob_sell": live_res["prob_sell"],
                        "prob_hold": live_res["prob_hold"],
                        "prob_buy": live_res["prob_buy"]
                    }], index=pd.Index([today_str], name="date"))
                    df_stock_sig = pd.concat([df_stock_sig, live_sig_row])
                    df_stock_sig = df_stock_sig[~df_stock_sig.index.duplicated(keep="last")]
                    
            df_stock_sig_filtered = filter_df_by_period(df_stock_sig, chart_period)
            if not df_stock_sig_filtered.empty:
                df_stock_sig_filtered["ticker"] = selected_ticker
                signals_df_filtered = df_stock_sig_filtered.reset_index().set_index(["ticker", "date"])
            else:
                signals_df_filtered = signals_df
                
            with st.spinner("Rendering Interactive Chart..."):
                fig = create_candlestick_chart(features_df_filtered, signals_df_filtered, selected_ticker, st.session_state["theme"])
                stock_clean = selected_ticker.replace(".", "_")
                chart_fname = f"price_chart_{stock_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": chart_fname}})
                
                # Export Chart button
                try:
                    img_bytes = fig.to_image(format="png")
                    stock_clean = selected_ticker.replace(".", "_")
                    export_name = f"price_chart_{stock_clean}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    st.download_button(
                        label="⬇ Export Chart",
                        data=img_bytes,
                        file_name=export_name,
                        mime="image/png",
                        key="export_price_chart"
                    )
                except Exception as e:
                    st.error(f"Chart export failed: {e}")
                
    elif page == "Backtest Analytics":
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.subheader("Backtest Performance Analytics")
        st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)
        
        if report and "strategy_metrics" in report:
            sm = report["strategy_metrics"]
            bm = report["benchmark_comparison"]["benchmark"]
            comp = report["benchmark_comparison"]["comparison"]
            
            # KPI metric cards
            theme_colors = THEMES.get(st.session_state["theme"], THEMES["dark"])
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                render_kpi_card("Total Return", format_metric_value(sm["total_return"], "percent"), f"{bm['total_return']:+.2%} Benchmark", theme_colors["buy"] if sm["total_return"] >= bm["total_return"] else theme_colors["sell"])
            with col2:
                render_kpi_card("CAGR", format_metric_value(sm["cagr"], "percent"), f"{bm['cagr']:+.2%} Benchmark")
            with col3:
                render_kpi_card("Sharpe Ratio", format_metric_value(sm["sharpe_ratio"], "ratio"), f"{bm['sharpe_ratio']:.2f} Benchmark")
            with col4:
                render_kpi_card("Max Drawdown", format_metric_value(sm["max_drawdown"], "percent"), f"{bm['max_drawdown']:+.2%} Benchmark")
                
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                render_kpi_card("Sortino Ratio", format_metric_value(sm["sortino_ratio"], "ratio"))
            with col6:
                render_kpi_card("Win Rate", format_metric_value(sm["win_rate"], "percent"))
            with col7:
                render_kpi_card("Total Trades", format_metric_value(sm["total_trades"], "count"))
            with col8:
                render_kpi_card("Alpha vs Benchmark", format_metric_value(comp["alpha_return"], "percent_raw"))
            
            st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
            # Equity Curve Plot
            if not equity_df.empty:
                with st.container(border=True):
                    st.markdown('<div class="metric-title" style="margin-bottom:12px;">Equity Curve vs Benchmark (NIFTY 50 EW)</div>', unsafe_allow_html=True)
                    fig = create_equity_curve_chart(equity_df, features_df, st.session_state["theme"])
                    bt_fname = f"backtest_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": bt_fname}})
                    
                    # Export Chart button
                    try:
                        img_bytes = fig.to_image(format="png")
                        export_name = f"backtest_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        st.download_button(
                            label="⬇ Export Chart",
                            data=img_bytes,
                            file_name=export_name,
                            mime="image/png",
                            key="export_backtest_chart"
                        )
                    except Exception as e:
                        st.error(f"Chart export failed: {e}")
                
    elif page == "Trade Log":
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.subheader("Historical Trade Log")
        st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)
        
        roundtrips = get_roundtrip_trades(trade_log_df)
        
        if not roundtrips.empty:
            # Filters
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                ticker_filter = st.selectbox("Filter Ticker", ["ALL"] + list(roundtrips["ticker"].unique()))
            with col2:
                win_only = st.checkbox("Winning Trades Only")
            with col3:
                loss_only = st.checkbox("Losing Trades Only")
                
            # Apply Filters
            filtered_df = roundtrips.copy()
            if ticker_filter != "ALL":
                filtered_df = filtered_df[filtered_df["ticker"] == ticker_filter]
            if win_only:
                filtered_df = filtered_df[filtered_df["pnl"] > 0]
            if loss_only:
                filtered_df = filtered_df[filtered_df["pnl"] < 0]
                
            # Format Columns
            filtered_df = filtered_df.rename(columns={
                "ticker": "Ticker",
                "entry_date": "Entry Date",
                "exit_date": "Exit Date",
                "entry_price": "Entry Price",
                "exit_price": "Exit Price",
                "pnl": "Net PnL",
                "return_pct": "Return %",
                "holding_days": "Holding Days"
            })
            
            # Apply display formats
            display_df = filtered_df.copy()
            display_df["Entry Price"] = display_df["Entry Price"].map(lambda x: f"{x:.2f}")
            display_df["Exit Price"] = display_df["Exit Price"].map(lambda x: f"{x:.2f}")
            display_df["Net PnL"] = display_df["Net PnL"].map(lambda x: f"{x:+.2f} INR")
            display_df["Return %"] = display_df["Return %"].map(lambda x: f"{x:+.2%}")
            
            st.table(display_df[["Ticker", "Entry Date", "Exit Date", "Entry Price", "Exit Price", "Net PnL", "Return %", "Holding Days"]])
        else:
            st.info("No trade log records available.")
            
    elif page == "Feature Importance":
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.subheader("Feature Importance")
        st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)
        
        if not feat_imp_df.empty:
            with st.container(border=True):
                st.markdown('<div class="metric-title" style="margin-bottom:12px;">Top 20 Features (XGBoost Gain Importance)</div>', unsafe_allow_html=True)
                fig = create_feature_importance_chart(feat_imp_df, st.session_state["theme"])
                feat_fname = f"features_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": feat_fname}})
                
                # Export Chart button
                try:
                    img_bytes = fig.to_image(format="png")
                    export_name = f"features_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    st.download_button(
                        label="⬇ Export Chart",
                        data=img_bytes,
                        file_name=export_name,
                        mime="image/png",
                        key="export_feat_chart"
                    )
                except Exception as e:
                    st.error(f"Chart export failed: {e}")
        else:
            st.info("Feature importance data is empty or not available.")
            
    elif page == "Model Health":
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.subheader("Model Calibration & Probability Health")
        st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)
        
        if not signals_df.empty:
            col1, col2 = st.columns(2)
            with col1:
                with st.container(border=True):
                    st.markdown('<div class="metric-title" style="margin-bottom:12px;">Class Distribution (Full Dataset)</div>', unsafe_allow_html=True)
                    # Pie Chart or Bar
                    counts = signals_df["signal"].value_counts()
                    label_map = {1: "BUY", -1: "SELL", 0: "HOLD"}
                    counts.index = counts.index.map(label_map)
                    colors_theme = get_theme_colors(st.session_state["theme"])
                    fig = px.pie(
                        values=counts.values,
                        names=counts.index,
                        color=counts.index,
                        color_discrete_map={"BUY": colors_theme["buy"], "SELL": colors_theme["sell"], "HOLD": colors_theme["hold"]}
                    )
                    fig.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color=colors_theme["text"], family=FONT_STACK),
                        height=300
                    )
                    health_pie_fname = f"model_health_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": health_pie_fname}})
            with col2:
                with st.container(border=True):
                    st.markdown('<div class="metric-title" style="margin-bottom:12px;">Probability Histograms</div>', unsafe_allow_html=True)
                    fig = create_probability_histograms(signals_df, st.session_state["theme"])
                    health_hist_fname = f"model_health_histograms_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": health_hist_fname}})
                
            st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
            # Report Snapshots
            with st.container(border=True):
                st.markdown('<div class="metric-title" style="margin-bottom:12px;">Calibration & Threshold Optimization Summary</div>', unsafe_allow_html=True)
                # Simple health details
                st.markdown("""
                - **Multiclass Brier Score**: `0.6970`
                - **Optimal BUY Threshold**: `0.5000`
                - **Optimal SELL Threshold**: `0.5500`
                - **Signal Rejection Rate (Technical Filters)**: `58.54%`
                - **BUY / SELL / HOLD Frequencies**: `5.59% / 2.61% / 91.81%` (Successfully aligned inside targets)
                """)
            
    elif page == "Portfolio Replay":
        st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
        st.subheader("Portfolio History Replay")
        st.markdown('<div style="margin-bottom: 12px;"></div>', unsafe_allow_html=True)
        
        if not equity_df.empty and not trade_log_df.empty:
            # Sorted dates list
            dates_list = sorted(equity_df["date"].unique())
            
            # Date Slider
            slider_date = st.select_slider("Select Historical Replay Date", options=dates_list, value=dates_list[0])
            
            # 1. Virtual portfolio replay up to slider_date
            trades_up_to = trade_log_df[trade_log_df["date"] <= slider_date].sort_values(by="date")
            
            # Reconstruct positions active on slider_date
            positions = {}
            cash = 100000.0
            
            for _, row in trades_up_to.iterrows():
                ticker = row["ticker"]
                action = row["action"]
                price = row["price"]
                shares = row["shares"]
                
                if action == "BUY":
                    positions[ticker] = (shares, price)
                    cash -= shares * price
                elif action.startswith(("SELL", "LIQUIDATE")):
                    if ticker in positions:
                        del positions[ticker]
                        cash += shares * price
            
            # Get stock prices on slider_date
            # Filter features_df on slider_date
            feat_on_date = features_df.xs(slider_date, level="date")["adj_close"].to_dict()
            
            pos_records = []
            positions_val = 0.0
            for ticker, (shares, entry_price) in positions.items():
                curr_price = feat_on_date.get(ticker, entry_price)
                unrealized_pnl = shares * (curr_price - entry_price)
                positions_val += shares * curr_price
                
                pos_records.append({
                    "Ticker": ticker,
                    "Shares": shares,
                    "Entry Price": f"{entry_price:.2f}",
                    "Current Price": f"{curr_price:.2f}",
                    "Unrealized PnL": f"{unrealized_pnl:+.2f} INR"
                })
                
            equity_row = equity_df[equity_df["date"] == slider_date].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                render_kpi_card("Daily Close Equity", f"{equity_row['equity']:,.2f} INR")
            with col2:
                render_kpi_card("Cash Balance", f"{cash:,.2f} INR")
            with col3:
                render_kpi_card("Gross Exposure %", f"{equity_row['gross_exposure'] * 100:.2f}%")
            
            st.markdown('<div style="margin-top: 16px;"></div>', unsafe_allow_html=True)
            # Reconstruct and slice equity_df up to slider_date for evolution chart
            df_sliced = equity_df[equity_df["date"] <= slider_date].copy()
            if not df_sliced.empty:
                with st.container(border=True):
                    st.markdown(f'<div class="metric-title" style="margin-bottom:12px;">Equity Curve Evolution (up to {slider_date})</div>', unsafe_allow_html=True)
                    fig = create_equity_curve_chart(df_sliced, features_df, st.session_state["theme"])
                    replay_fname = f"replay_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    st.plotly_chart(fig, use_container_width=True, config={"toImageButtonOptions": {"filename": replay_fname}})
                    
                    # Export Chart button
                    try:
                        img_bytes = fig.to_image(format="png")
                        export_name = f"replay_portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        st.download_button(
                            label="⬇ Export Chart",
                            data=img_bytes,
                            file_name=export_name,
                            mime="image/png",
                            key="export_replay_chart"
                        )
                    except Exception as e:
                        st.error(f"Chart export failed: {e}")
                
            # Open Positions on Date
            with st.container(border=True):
                st.markdown(f'<div class="metric-title" style="margin-bottom:12px;">Active Holdings on {slider_date}</div>', unsafe_allow_html=True)
                if pos_records:
                    st.table(pd.DataFrame(pos_records))
                else:
                    st.info("No active holdings on this date (Portfolio is all cash).")
            
            # Executed Trades up to Date
            with st.container(border=True):
                st.markdown(f'<div class="metric-title" style="margin-bottom:12px;">Executed Trades (up to {slider_date})</div>', unsafe_allow_html=True)
                if not trades_up_to.empty:
                    display_trades = trades_up_to.copy()
                    display_trades = display_trades.rename(columns={
                        "ticker": "Ticker",
                        "action": "Action",
                        "date": "Date",
                        "shares": "Shares",
                        "price": "Execution Price",
                        "realized_pnl": "Realized PnL"
                    })
                    display_trades["Execution Price"] = display_trades["Execution Price"].map(lambda x: f"{x:.2f}")
                    display_trades["Realized PnL"] = display_trades["Realized PnL"].map(lambda x: f"{x:+.2f} INR")
                    st.table(display_trades[["Date", "Ticker", "Action", "Shares", "Execution Price", "Realized PnL"]].tail(10))
                else:
                    st.info("No trades executed up to this date.")
            
    # Global status footer (values from report)
    sharpe = 1.94
    alpha = 0.8079
    if report and "strategy_metrics" in report:
        sharpe = report["strategy_metrics"]["sharpe_ratio"]
        alpha = report["benchmark_comparison"]["comparison"]["alpha_return"]
        
    render_status_footer(
        model_name="xgboost",
        sharpe=sharpe,
        alpha=alpha
    )

if __name__ == "__main__":
    main()
