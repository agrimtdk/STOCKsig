import streamlit as st
import logging

logger = logging.getLogger(__name__)

FONT_STACK = "'JetBrains Mono', 'IBM Plex Mono', 'SF Mono', monospace"

THEMES = {
    "dark": {
        "background": "#121212",
        "card": "#1E1E1E",
        "border": "#2E2E2E",
        "text": "#E6EDF3",
        "muted": "#8B949E",
        "accent": "#E5E7EB",
        "buy": "#22C55E",
        "sell": "#EF4444",
        "hold": "#6B7280",
        "bull_candle": "#26A69A",
        "bear_candle": "#EF5350",
        "sma": "#A3A3A3",
        "ema": "#F59E0B",
        "badge_bg": "rgba(46, 46, 46, 0.4)",
        "hover": "#2D2D2D",
        "grid": "rgba(46, 46, 46, 0.15)"
    },
    "light": {
        "background": "#F8FAFC",
        "card": "#FFFFFF",
        "border": "#CBD5E1",
        "text": "#0F172A",
        "muted": "#475569",
        "accent": "#2563EB",
        "buy": "#16A34A",
        "sell": "#DC2626",
        "hold": "#64748B",
        "bull_candle": "#16A34A",
        "bear_candle": "#DC2626",
        "sma": "#2563EB",
        "ema": "#D97706",
        "badge_bg": "rgba(203, 213, 225, 0.4)",
        "hover": "#F1F5F9",
        "grid": "rgba(203, 213, 225, 0.15)"
    }
}

def inject_global_styles(theme_name: str):
    """
    Injects custom Vanilla CSS to style the Streamlit app like a monospace quant terminal.
    Dynamically switches colors based on the selected theme.
    """
    theme = THEMES.get(theme_name, THEMES["dark"])
    bg = theme["background"]
    card_bg = theme["card"]
    border = theme["border"]
    text = theme["text"]
    muted = theme["muted"]
    hover = theme["hover"]
    accent = theme["accent"]
    buy = theme["buy"]
    sell = theme["sell"]
    hold = theme["hold"]
    badge_bg = theme["badge_bg"]
        
    css = f"""
    <style>
    /* 1. Global typography and background */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"], .stText, .stMarkdown, .stButton, div, span, table {{
        font-family: {FONT_STACK} !important;
    }}
    
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stMainBlockContainer"], [data-testid="block-container"], div[data-testid="stVerticalBlock"] {{
        background-color: {bg} !important;
        color: {text} !important;
    }}
    
    /* Hide Streamlit default headers/footers for terminal look */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Permanently hide sidebar and reclaim left space */
    section[data-testid="stSidebar"] {{
        display: none !important;
    }}
    [data-testid="stSidebarCollapsedControl"] {{
        display: none !important;
    }}
    div[data-testid="stAppViewContainer"] {{
        margin-left: 0rem !important;
        padding-left: 0rem !important;
    }}
    [data-testid="stMainView"] {{
        margin-left: 0px !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        left: 0px !important;
        width: 100% !important;
    }}
    [data-testid="block-container"] {{
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }}
    
    /* Sticky Top Navbar configuration for block-container's first child */
    div[data-testid="block-container"] > div:first-child {{
        position: -webkit-sticky;
        position: sticky;
        top: 0;
        background-color: {bg} !important;
        z-index: 9999;
        padding-top: 6px !important;
        padding-bottom: 6px !important;
        border-bottom: 1px solid {border} !important;
        margin-bottom: 10px !important;
    }}
    
    /* 2. Custom outlined cards & native streamlit containers styling */
    .quant-card {{
        background-color: {card_bg};
        border: 1px solid {border};
        border-radius: 4px;
        padding: 12px;
        margin-bottom: 8px;
        color: {text};
        transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .quant-card:hover {{
        background-color: {hover};
        border-color: {accent} !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.15);
    }}
    
    /* Target Streamlit's native st.container(border=True) and border wrappers */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderDiv"] {{
        background-color: {card_bg} !important;
        background: {card_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 4px !important;
        padding: 14px !important;
        margin-bottom: 12px !important;
        transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
    }}
    div[data-testid="stVerticalBlockBorderDiv"]:hover {{
        background-color: {hover} !important;
        border-color: {accent} !important;
        box-shadow: 0 0 10px rgba(59, 130, 246, 0.15) !important;
    }}
    
    /* Target Signal Glows on st.container(border=True) using parent selector */
    div[data-testid="stVerticalBlockBorderDiv"]:has(.border-buy):hover {{
        border-color: {buy} !important;
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.2) !important;
    }}
    div[data-testid="stVerticalBlockBorderDiv"]:has(.border-sell):hover {{
        border-color: {sell} !important;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.2) !important;
    }}
    div[data-testid="stVerticalBlockBorderDiv"]:has(.border-hold):hover {{
        border-color: {hold} !important;
        box-shadow: 0 0 12px rgba(107, 114, 128, 0.15) !important;
    }}
    
    /* Left border highlights */
    .border-buy {{ border-left: 4px solid {buy} !important; }}
    .border-sell {{ border-left: 4px solid {sell} !important; }}
    .border-hold {{ border-left: 4px solid {hold} !important; }}
    
    /* 3. Terminal metrics and labels */
    .metric-title {{
        font-size: 11px;
        color: {muted};
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        font-weight: 700;
    }}
    .metric-value {{
        font-size: 22px;
        font-weight: 700;
        color: {text};
        margin-bottom: 2px;
    }}
    .metric-change {{
        font-size: 10px;
        font-weight: 500;
    }}
    
    /* Consistent spacing between vertical blocks */
    div[data-testid="stVerticalBlock"] {{
        gap: 0.75rem !important;
    }}
    
    /* Subheader spacing */
    h2, h3, [data-testid="stSubheader"] {{
        margin-top: 0.25rem !important;
        margin-bottom: 0.5rem !important;
        padding-bottom: 8px !important;
        border-bottom: 1px solid {border} !important;
        color: {text} !important;
        font-family: {FONT_STACK} !important;
        letter-spacing: 0.04em !important;
    }}
    
    /* Horizontal column gap */
    div[data-testid="stHorizontalBlock"] {{
        gap: 0.75rem !important;
    }}
    
    /* 4. Badges and pills */
    .badge, .badge-buy, .badge-sell, .badge-hold, .badge-risk-low, .badge-risk-medium, .badge-risk-high, .sentiment-badge, .quant-badge {{
        display: inline-block;
        padding: 2px 6px;
        border-radius: 3px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        background-color: {badge_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}
    
    /* 5. Compact tables */
    .dataframe {{
        font-size: 11px !important;
        border-collapse: collapse !important;
        border: 1px solid {border} !important;
        width: 100% !important;
    }}
    .dataframe th {{
        background-color: {card_bg} !important;
        color: {muted} !important;
        border: 1px solid {border} !important;
        padding: 6px !important;
        text-transform: uppercase !important;
    }}
    .dataframe td {{
        border: 1px solid {border} !important;
        padding: 6px !important;
    }}
    
    /* 6. Sticky Terminal Footer */
    .status-footer {{
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: {card_bg};
        border-top: 1px solid {border};
        padding: 4px 16px;
        font-size: 9px;
        color: {muted};
        z-index: 999999;
        display: flex;
        justify-content: space-between;
    }}
    .status-footer span {{
        margin-right: 16px;
    }}
    
    .disclaimer-footer {{
        position: fixed;
        bottom: 19px; /* Sits immediately above the 19px tall status-footer */
        left: 0;
        width: 100%;
        background-color: {bg};
        border-top: 1px solid #202938;
        padding: 8px 12px;
        font-size: 8.5px;
        font-family: {FONT_STACK} !important;
        color: #8B949E;
        text-align: center;
        opacity: 0.8;
        z-index: 999998;
        line-height: 1.4;
    }}
    
    /* Padding for footer offset to prevent layout overlap */
    .footer-offset {{
        margin-bottom: 110px;
    }}
    
    /* 7. Selectbox and dropdown theme mapping */
    div[data-baseweb="select"] {{
        background: {card_bg} !important;
        background-color: {card_bg} !important;
    }}
    div[data-baseweb="select"] input {{
        color: {text} !important;
    }}
    div[data-baseweb="select"] svg {{
        color: {muted} !important;
    }}
    div[data-baseweb="select"] > div {{
        background-color: {card_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}
    /* Dropdown menu items */
    div[role="listbox"], ul[role="listbox"] {{
        background-color: {card_bg} !important;
        color: {text} !important;
        border: 1px solid {border} !important;
    }}
    li[role="option"] {{
        background-color: {card_bg} !important;
        color: {text} !important;
    }}
    li[role="option"]:hover, li[role="option"][aria-selected="true"] {{
        background-color: {hover} !important;
        color: {text} !important;
    }}
    
    /* 8. Glossary Hover Tooltips */
    .glossary-tooltip {{
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dotted {muted} !important;
    }}
    .glossary-tooltip .glossary-tooltiptext {{
        visibility: hidden;
        opacity: 0;
        width: 220px;
        background: rgba(18, 24, 33, 0.90) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #E5E7EB !important;
        text-align: left;
        border-radius: 10px;
        padding: 10px;
        position: absolute;
        z-index: 999999999 !important;
        bottom: 125%; /* Position above the text */
        left: 50%;
        transform: translateX(-50%);
        transition: opacity 0.15s ease-in-out, visibility 0.15s ease-in-out;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        line-height: 1.4 !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        pointer-events: none; /* Avoid blocking hover pointer */
        white-space: normal;
    }}
    .glossary-tooltip:hover .glossary-tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def get_navbar_css(active_index: int, theme_name: str) -> str:
    """
    Generates dynamic CSS for the top horizontal navbar to style active tabs.
    Supports responsive horizontal scrolling.
    """
    css = f"""
    <style>
    /* Styling all buttons inside the first horizontal row (the navbar block) */
    div[data-testid="block-container"] > div:first-child button {{
        background-color: #1E1E1E !important;
        background: #1E1E1E !important;
        border: 1px solid #2E2E2E !important;
        box-shadow: none !important;
        color: #E6EDF3 !important;
        font-family: {FONT_STACK} !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        padding: 8px 14px !important;
        border-radius: 8px !important;
        border-bottom: 2px solid transparent !important;
        transition: color 0.15s ease, border-color 0.15s ease, background-color 0.15s ease;
    }}
    
    div[data-testid="block-container"] > div:first-child button:hover {{
        color: #E6EDF3 !important;
        background-color: #2D2D2D !important;
        background: #2D2D2D !important;
        border: 1px solid #2E2E2E !important;
        box-shadow: none !important;
    }}
    
    div[data-testid="block-container"] > div:first-child button:focus,
    div[data-testid="block-container"] > div:first-child button:active {{
        color: #E6EDF3 !important;
        background-color: #1E1E1E !important;
        background: #1E1E1E !important;
        border: 1px solid #2E2E2E !important;
        box-shadow: none !important;
    }}
    
    /* Active tab style with custom border-bottom highlight, targeting the active nested column inside col_nav */
    div[data-testid="block-container"] > div:first-child div[data-testid="column"]:nth-child(2) div[data-testid="column"]:nth-child({active_index}) button {{
        color: #E6EDF3 !important;
        font-weight: 700 !important;
        border-bottom: 2px solid #E5E7EB !important;
    }}
    </style>
    """
    return css

def render_signal_card(
    ticker: str,
    signal_label: str,
    confidence: float,
    probs: dict,
    theme: str
):
    """
    Renders Page 1 Outlined signal card details inside a native card container.
    """
    from src.dashboard.glossary import tooltip
    border_class = f"border-{signal_label.lower()}"
    badge_class = f"badge-{signal_label.lower()}"
    
    ticker_title = f"{ticker} Latest Signal"
    conf_label = f"Conf: {confidence:.1f}%"
    conf_html = tooltip("Signal Confidence", conf_label)
    
    st.markdown(f"""
    <div class="{border_class}" style="padding-left: 8px;">
        <div class="metric-title">{ticker_title}</div>
        <div style="display: flex; align-items: baseline; margin-top: 6px;">
            <span style="font-size: 28px; font-weight: 700; margin-right: 12px; line-height: 1;">{signal_label}</span>
            <span class="badge {badge_class}">{conf_html}</span>
        </div>
        <div style="margin-top: 10px; font-size: 10px;">
            <span style="margin-right: 12px;">P(BUY): {probs.get('prob_buy', 0.0):.2f}</span>
            <span style="margin-right: 12px;">P(HOLD): {probs.get('prob_hold', 0.0):.2f}</span>
            <span>P(SELL): {probs.get('prob_sell', 0.0):.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_card(title: str, value: str, change: str = None, color: str = "inherit"):
    """
    Renders a standard KPI card with glossary hover tooltips.
    """
    from src.dashboard.glossary import tooltip
    title_html = title
    for term in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "CAGR", "Alpha", "Beta", "Win Rate", "Profit Factor", "Expectancy", "Volatility"]:
        if term in title:
            title_html = title.replace(term, tooltip(term))
            break
            
    change_html = ""
    if change:
        import re
        match = re.search(r"[-+]?\d*\.?\d+", change)
        num = float(match.group()) if match else 0.0
        sign = "+" if num >= 0 else ""
        change_html = f'<div class="metric-change" style="color: {color};">{sign}{change}</div>'
        
    st.markdown(f"""
    <div class="quant-card">
        <div class="metric-title">{title_html}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

def render_status_footer(model_name: str, sharpe: float, alpha: float):
    """
    Renders a sticky monospace footer showing critical system details and a professional disclaimer.
    """
    st.markdown(f"""
    <div class="disclaimer-footer">
        <div style="font-weight: 700; margin-bottom: 3px; letter-spacing: 0.05em;">DISCLAIMER</div>
        STOCKsig is an independent research and educational side project built to explore quantitative analysis, machine learning, and financial signal engineering.
        The insights, analytics, and market signals presented are intended for informational purposes only and should not be interpreted as financial or investment advice.
        Financial markets are subject to risk and volatility. Any investment decisions made using this platform or its outputs are solely at the user’s discretion and risk.
        Please conduct your own research and consult a qualified financial advisor before making investment decisions.
    </div>
    <div class="status-footer">
        <div>
            <span>SYSTEM STATE: ACTIVE</span>
            <span>MODEL: {model_name.upper()}</span>
        </div>
        <div>
            <span>SHARPE: {sharpe:.2f}</span>
            <span>ALPHA vs BENCHMARK: {alpha:+.2%}</span>
        </div>
    </div>
    <div class="footer-offset"></div>
    """, unsafe_allow_html=True)
