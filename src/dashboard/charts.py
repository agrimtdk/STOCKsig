import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

from src.dashboard.components import FONT_STACK, THEMES

def get_theme_colors(theme_name: str) -> dict:
    """
    Returns layout colors based on the active theme ('dark' or 'light').
    """
    theme = THEMES.get(theme_name, THEMES["dark"])
    return {
        "bg": theme["background"],
        "paper": theme["card"],
        "card": theme["card"],
        "border": theme["border"],
        "text": theme["text"],
        "secondary_text": theme["muted"],
        "accent": theme["accent"],
        "buy": theme["buy"],
        "sell": theme["sell"],
        "hold": theme["hold"],
        "bull": theme["bull_candle"],
        "bear": theme["bear_candle"],
        "sma": theme["sma"],
        "ema": theme["ema"],
        "grid": theme["grid"]
    }

def create_candlestick_chart(
    features_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    ticker: str,
    theme: str
) -> go.Figure:
    """
    Creates Page 2 Candlestick chart for the selected stock with BUY/SELL overlays,
    SMA_20, EMA_20, and confidence heatmap markers.
    """
    colors = get_theme_colors(theme)
    
    # Filter ticker data
    df_stock = features_df.xs(ticker, level="ticker").copy()
    df_sig = signals_df.xs(ticker, level="ticker").copy()
    
    # Ensure they are sorted by date
    df_stock = df_stock.sort_index()
    df_sig = df_sig.sort_index()
    
    # Align dates
    common_dates = df_stock.index.intersection(df_sig.index)
    df_stock = df_stock.loc[common_dates]
    df_sig = df_sig.loc[common_dates]
    
    fig = go.Figure()
    
    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df_stock.index,
        open=df_stock["open"],
        high=df_stock["high"],
        low=df_stock["low"],
        close=df_stock["close"],
        name="OHLC",
        increasing_line_color=colors["bull"],
        decreasing_line_color=colors["bear"],
        increasing_fillcolor=colors["bull"],
        decreasing_fillcolor=colors["bear"],
        line_width=1.0
    ))
    
    # 2. Indicators (SMA_20 & EMA_20)
    if "sma_20" in df_stock.columns:
        fig.add_trace(go.Scatter(
            x=df_stock.index,
            y=df_stock["sma_20"],
            name="SMA 20",
            line=dict(color=colors["sma"], width=1.0),
            opacity=0.8
        ))
    if "ema_20" in df_stock.columns:
        fig.add_trace(go.Scatter(
            x=df_stock.index,
            y=df_stock["ema_20"],
            name="EMA 20",
            line=dict(color=colors["ema"], width=1.0, dash="dash"),
            opacity=0.8
        ))
        
    # 3. Buy/Sell confidence heatmap markers
    # Filter non-zero signals
    buys = df_sig[df_sig["signal"] == 1]
    sells = df_sig[df_sig["signal"] == -1]
    
    if not buys.empty:
        buy_prices = df_stock.loc[buys.index, "low"] * 0.98
        fig.add_trace(go.Scatter(
            x=buys.index,
            y=buy_prices,
            mode="markers",
            name="BUY Signals",
            opacity=0.7,
            marker=dict(
                symbol="triangle-up",
                size=8,
                color=buys["confidence_score"],
                colorscale="Greens",
                showscale=True,
                colorbar=dict(
                    title=dict(text="BUY Conf", font=dict(family=FONT_STACK, size=10, color=colors["text"])),
                    tickfont=dict(family=FONT_STACK, size=8, color=colors["text"]),
                    thickness=10,
                    len=0.4,
                    y=0.8
                )
            ),
            text=[f"Conf: {c:.1f}%" for c in buys["confidence_score"]],
            hoverinfo="x+y+text"
        ))
        
    if not sells.empty:
        sell_prices = df_stock.loc[sells.index, "high"] * 1.02
        fig.add_trace(go.Scatter(
            x=sells.index,
            y=sell_prices,
            mode="markers",
            name="SELL Signals",
            opacity=0.7,
            marker=dict(
                symbol="triangle-down",
                size=8,
                color=sells["confidence_score"],
                colorscale="Reds",
                showscale=True,
                colorbar=dict(
                    title=dict(text="SELL Conf", font=dict(family=FONT_STACK, size=10, color=colors["text"])),
                    tickfont=dict(family=FONT_STACK, size=8, color=colors["text"]),
                    thickness=10,
                    len=0.4,
                    y=0.3
                )
            ),
            text=[f"Conf: {c:.1f}%" for c in sells["confidence_score"]],
            hoverinfo="x+y+text"
        ))
        
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        plot_bgcolor=colors["bg"],
        paper_bgcolor=colors["paper"],
        font=dict(color=colors["text"], family=FONT_STACK),
        xaxis=dict(
            gridcolor=colors["grid"],
            linecolor=colors["border"],
            showline=True,
            rangeslider=dict(visible=False),
            type="category",  # removes weekend gaps
            tickmode="auto",
            nticks=10
        ),
        yaxis=dict(
            gridcolor=colors["grid"],
            linecolor=colors["border"],
            showline=True,
            side="right"
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.05, x=0.01, font=dict(family=FONT_STACK, size=10, color=colors["text"])),
        height=600
    )
    return fig

def create_equity_curve_chart(equity_df: pd.DataFrame, features_df: pd.DataFrame, theme: str) -> go.Figure:
    """
    Creates Page 3 Equity curve vs Benchmark comparison chart.
    """
    colors = get_theme_colors(theme)
    fig = go.Figure()
    
    # Scale both to start at 100k
    strategy_equity = equity_df["equity"]
    
    fig.add_trace(go.Scatter(
        x=equity_df["date"],
        y=strategy_equity,
        name="Strategy (STOCKsig)",
        line=dict(color=colors["buy"], width=2.0)
    ))
    
    # Reconstruct/Scale benchmark curve from features_df
    if not features_df.empty:
        try:
            df_feat = features_df.copy()
            if "ticker" in df_feat.index.names and "date" in df_feat.index.names:
                df_feat = df_feat.reset_index()
                
            df_feat["date"] = df_feat["date"].astype(str)
            start_date = str(equity_df["date"].iloc[0])
            end_date = str(equity_df["date"].iloc[-1])
            df_feat = df_feat[(df_feat["date"] >= start_date) & (df_feat["date"] <= end_date)]
            
            # Pivot to get adj_close for each ticker
            df_pivot = df_feat.pivot(index="date", columns="ticker", values="adj_close").sort_index()
            df_pivot = df_pivot.ffill().bfill()
            
            if not df_pivot.empty:
                first_row = df_pivot.iloc[0]
                shares = 10000.0 / first_row
                bench_equity = df_pivot.multiply(shares).sum(axis=1)
                
                fig.add_trace(go.Scatter(
                    x=bench_equity.index,
                    y=bench_equity.values,
                    name="Benchmark (Equal-Weighted)",
                    line=dict(color=colors["hold"], width=1.5, dash="dash")
                ))
        except Exception as e:
            logger.error(f"Error building benchmark in chart: {e}")
            
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        plot_bgcolor=colors["bg"],
        paper_bgcolor=colors["paper"],
        font=dict(color=colors["text"], family=FONT_STACK),
        xaxis=dict(gridcolor=colors["grid"], linecolor=colors["border"], showline=True),
        yaxis=dict(gridcolor=colors["grid"], linecolor=colors["border"], showline=True, side="right"),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.05, x=0.01, font=dict(family=FONT_STACK, size=10, color=colors["text"])),
        height=450
    )
    return fig

def create_feature_importance_chart(importance_df: pd.DataFrame, theme: str) -> go.Figure:
    """
    Creates Page 5 Feature Importance horizontal bar chart.
    """
    colors = get_theme_colors(theme)
    if importance_df.empty:
        return go.Figure()
        
    df_sorted = importance_df.sort_values(by="mean_importance", ascending=True).tail(20)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["feature"],
        x=df_sorted["mean_importance"],
        orientation="h",
        marker_color=colors["accent"],
        name="Mean Importance"
    ))
    
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        plot_bgcolor=colors["bg"],
        paper_bgcolor=colors["paper"],
        font=dict(color=colors["text"], family=FONT_STACK),
        xaxis=dict(gridcolor=colors["grid"], linecolor=colors["border"], showline=True),
        yaxis=dict(gridcolor=colors["grid"], linecolor=colors["border"], showline=True),
        margin=dict(l=150, r=20, t=40, b=20),
        height=500
    )
    return fig

def create_sparkline(prices: pd.Series, theme: str) -> go.Figure:
    """
    Creates a tiny, axis-free sparkline for the latest signal card.
    """
    colors = get_theme_colors(theme)
    fig = go.Figure()
    
    # Determine color based on overall direction
    line_color = colors["buy"] if prices.iloc[-1] >= prices.iloc[0] else colors["sell"]
    
    fig.add_trace(go.Scatter(
        x=list(range(len(prices))),
        y=prices,
        mode="lines",
        line=dict(color=line_color, width=2.0),
        hoverinfo="none"
    ))
    
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=5, r=5, t=5, b=5),
        height=65,
        showlegend=False
    )
    return fig

def create_probability_histograms(signals_df: pd.DataFrame, theme: str) -> go.Figure:
    """
    Creates Page 6 multi-class probability histograms.
    """
    colors = get_theme_colors(theme)
    fig = go.Figure()
    
    if "prob_buy" in signals_df.columns:
        fig.add_trace(go.Histogram(
            x=signals_df["prob_buy"],
            name="BUY Prob",
            marker_color=colors["buy"],
            opacity=0.6,
            nbinsx=30
        ))
    if "prob_sell" in signals_df.columns:
        fig.add_trace(go.Histogram(
            x=signals_df["prob_sell"],
            name="SELL Prob",
            marker_color=colors["sell"],
            opacity=0.6,
            nbinsx=30
        ))
    if "prob_hold" in signals_df.columns:
        fig.add_trace(go.Histogram(
            x=signals_df["prob_hold"],
            name="HOLD Prob",
            marker_color=colors["hold"],
            opacity=0.4,
            nbinsx=30
        ))
        
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        barmode="overlay",
        plot_bgcolor=colors["bg"],
        paper_bgcolor=colors["paper"],
        font=dict(color=colors["text"], family=FONT_STACK),
        xaxis=dict(title="Probability", gridcolor=colors["grid"], linecolor=colors["border"]),
        yaxis=dict(title="Frequency", gridcolor=colors["grid"], linecolor=colors["border"]),
        margin=dict(l=50, r=20, t=40, b=40),
        legend=dict(orientation="h", y=1.05, x=0.01),
        height=350
    )
    return fig

def create_confidence_gauge(confidence: float, signal_label: str, theme: str) -> go.Figure:
    """
    Creates a radial Plotly gauge for the latest model confidence.
    """
    colors = get_theme_colors(theme)
    
    if signal_label == "BUY":
        color = colors["buy"]
    elif signal_label == "SELL":
        color = colors["sell"]
    else:
        color = colors["hold"]
        
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={
            "suffix": "%",
            "font": {"size": 20, "family": FONT_STACK, "color": colors["text"]}
        },
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": colors["border"],
                "tickfont": {"family": FONT_STACK, "size": 9}
            },
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": colors["border"],
            "steps": [{"range": [0, 100], "color": "rgba(0,0,0,0)"}]
        }
    ))
    
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"], family=FONT_STACK),
        margin=dict(l=10, r=10, t=5, b=5),
        height=65
    )
    return fig

def create_signal_history_timeline(signals_df: pd.DataFrame, ticker: str, latest_date, theme: str) -> go.Figure:
    """
    Creates a horizontal timeline showing the sequence and confidence of the last 10 signals.
    """
    colors = get_theme_colors(theme)
    
    # Filter ticker signals up to latest_date, take last 10
    df_sig = signals_df.xs(ticker, level="ticker").sort_index()
    df_10 = df_sig.loc[:latest_date].tail(10)
    
    # Convert dates to string for horizontal axis labels
    dates_str = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in df_10.index]
    
    # Map colors and labels
    signal_labels = []
    marker_colors = []
    sizes = []
    
    for _, row in df_10.iterrows():
        sig = row["signal"]
        conf = row.get("confidence_score", 0.0)
        if sig == 1:
            signal_labels.append(f"BUY ({conf:.1f}%)")
            marker_colors.append(colors["buy"])
            sizes.append(14)
        elif sig == -1:
            signal_labels.append(f"SELL ({conf:.1f}%)")
            marker_colors.append(colors["sell"])
            sizes.append(14)
        else:
            signal_labels.append(f"HOLD ({conf:.1f}%)")
            marker_colors.append(colors["hold"])
            sizes.append(10)
            
    fig = go.Figure()
    
    # Add horizontal line connecting markers
    fig.add_trace(go.Scatter(
        x=dates_str,
        y=[0] * len(df_10),
        mode="lines",
        line=dict(color=colors["border"], width=1.5),
        hoverinfo="none",
        showlegend=False
    ))
    
    # Add markers
    fig.add_trace(go.Scatter(
        x=dates_str,
        y=[0] * len(df_10),
        mode="markers+text",
        text=[l.split()[0] for l in signal_labels],  # BUY / SELL / HOLD
        textposition="top center",
        textfont=dict(family=FONT_STACK, size=8, color=colors["text"]),
        marker=dict(
            color=marker_colors,
            size=sizes,
            symbol="square",
            line=dict(color=colors["text"], width=1)
        ),
        hovertext=[f"{pd.to_datetime(d).strftime('%Y-%m-%d')}<br>Signal: {l}" for d, l in zip(df_10.index, signal_labels)],
        hoverinfo="text",
        showlegend=False
    ))
    
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"], family=FONT_STACK),
        xaxis=dict(
            visible=True,
            showgrid=False,
            showline=False,
            tickfont=dict(family=FONT_STACK, size=8, color=colors["text"]),
            ticks=""
        ),
        yaxis=dict(
            visible=False,
            range=[-1, 1]
        ),
        margin=dict(l=10, r=10, t=20, b=10),
        height=80,
        showlegend=False
    )
    return fig

def create_momentum_charts(features_df: pd.DataFrame, ticker: str, latest_date, theme: str) -> go.Figure:
    """
    Renders stacked subplots showing RSI (30d) and MACD Hist (30d) with monospace labels.
    """
    from plotly.subplots import make_subplots
    colors = get_theme_colors(theme)
    
    # Filter ticker data
    df_stock = features_df.xs(ticker, level="ticker").copy()
    df_stock = df_stock.sort_index()
    # Take last 30 trading days up to latest_date
    df_30d = df_stock.loc[:latest_date].tail(30)
    
    # Convert index dates to string for clean x-axis formatting
    dates_str = [pd.to_datetime(d).strftime("%Y-%m-%d") for d in df_30d.index]
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,
        subplot_titles=("RSI (14)", "MACD Histogram")
    )
    
    # 1. RSI trace
    if "rsi" in df_30d.columns:
        fig.add_trace(go.Scatter(
            x=dates_str,
            y=df_30d["rsi"],
            mode="lines",
            name="RSI",
            line=dict(color=colors["accent"], width=1.5),
            showlegend=False
        ), row=1, col=1)
        # Add 30 and 70 markers
        fig.add_hline(y=30, line_dash="dash", line_color=colors["hold"], opacity=0.4, row=1, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=colors["hold"], opacity=0.4, row=1, col=1)
        
    # 2. MACD hist trace
    if "macd_hist" in df_30d.columns:
        bar_colors = [colors["buy"] if v >= 0 else colors["sell"] for v in df_30d["macd_hist"]]
        fig.add_trace(go.Bar(
            x=dates_str,
            y=df_30d["macd_hist"],
            name="MACD Hist",
            marker_color=bar_colors,
            showlegend=False
        ), row=2, col=1)
        
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=colors["text"], family=FONT_STACK),
        margin=dict(l=25, r=10, t=15, b=10),
        height=140
    )
    
    # Update axes styling
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=colors["grid"], linecolor=colors["border"], tickfont=dict(family=FONT_STACK, size=8), row=i, col=1)
        fig.update_yaxes(gridcolor=colors["grid"], linecolor=colors["border"], tickfont=dict(family=FONT_STACK, size=8), row=i, col=1)
        
    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=9, family=FONT_STACK, color=colors["text"])
        
    return fig
