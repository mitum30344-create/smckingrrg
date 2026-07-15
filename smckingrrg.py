import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta

# Page configuration for wide fluid grid
st.set_page_config(page_title="StockMojo Sector Rotation RRG", layout="wide")

# Custom StockMojo Dark UI Stylesheet
st.markdown("""
    <style>
    .stApp { background-color: #0e1118; color: #cbd5e1; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    .stSelectbox, .stSlider { color: #ffffff !important; }
    div[data-testid="stDataFrame"] { background-color: #161b26 !important; border-radius: 6px; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 StockMojo Style Sector Rotation Graph (RRG)")

# --- 1. STOCKMOJO SECTOR INDICES MAP (NSE INDICES TICKERS) ---
sector_indices = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "AUTO": "^CNXAUTO",
    "ENERGY": "^CNXENERGY",
    "FMCG": "^CNXFMCG",
    "INFRA": "^CNXINFRA",
    "IT": "^CNXIT",
    "MEDIA": "^CNXMEDIA",
    "METAL": "^CNXMETAL",
    "PHARMA": "^CNXPHARMA",
    "PSU BANK": "^CNXPSUBANK",
    "PVT BANK": "^CNXPVTBANK",
    "REALTY": "^CNXREALTY",
    "SERVICES": "^CNXSERVICE"
}

all_index_tickers = list(sector_indices.values())
ticker_benchmark = "^CRSLDX" # Nifty 500 Benchmark

# --- 2. SIDEBAR ENGINE CONTROLS ---
st.sidebar.markdown("### 🛠️ RRG Settings")
timeframe_option = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly"])
base_tail_days = st.sidebar.slider("Tail Length", min_value=3, max_value=15, value=5)

st.sidebar.markdown("### 🔍 Zoom & Pan Control")
zoom_level = st.sidebar.slider("Zoom Scale Multiplier", min_value=1, max_value=10, value=4)
center_shift_x = st.sidebar.slider("Pan Axis (X)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
center_shift_y = st.sidebar.slider("Pan Axis (Y)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)

if timeframe_option == "Weekly":
    interval, days_back, pct_period = "1wk", 730, 4
else:
    interval, days_back, pct_period = "1d", 500, 5

end_date = datetime.today()
start_date = end_date - timedelta(days=days_back)

with st.spinner("Downloading Live NSE Sectoral Feeds..."):
    # Download single-level columns directly to avoid multi-index errors
    data = yf.download(all_index_tickers + [ticker_benchmark], start=start_date, end=end_date, interval=interval, group_by='ticker')

if not data.empty:
    # Safely parse close and volume across downloaded dictionary matrix
    close_prices = pd.DataFrame()
    volumes = pd.DataFrame()
    
    for t in all_index_tickers + [ticker_benchmark]:
        if t in data.columns.levels[0]:
            close_prices[t] = data[t]['Close']
            volumes[t] = data[t]['Volume']
            
    close_prices = close_prices.dropna(subset=[ticker_benchmark])
    volumes = volumes.loc[close_prices.index]
    
    # Accurate return calculation on 1D metrics
    pct_changes = close_prices[all_index_tickers].pct_change(periods=pct_period).iloc[-1] * 100

    # RRG Matrix Calculations for Whole Sectors
    rs_df = pd.DataFrame(index=close_prices.index)
    for ticker in all_index_tickers:
        rs_df[ticker] = close_prices[ticker] / close_prices[ticker_benchmark]

    rs_mean = rs_df.rolling(window=14).mean()
    rs_std = rs_df.rolling(window=14).std()
    rs_ratio = 100 + ((rs_df - rs_mean) / (rs_std + 1e-8)) * 1.2

    rs_roc = rs_df.pct_change(periods=5)
    vol_mean = volumes[all_index_tickers].rolling(window=14).mean()
    vol_std = volumes[all_index_tickers].rolling(window=14).std()
    norm_volume = (volumes[all_index_tickers] - vol_mean) / (vol_std + 1e-8)

    raw_momentum = rs_roc * np.tanh(norm_volume)
    mom_mean = raw_momentum.rolling(window=14).mean()
    mom_std = raw_momentum.rolling(window=14).std()
    rs_momentum = 100 + ((raw_momentum - mom_mean) / (mom_std + 1e-8)) * 1.2

    inv_sector_map = {v: k for k, v in sector_indices.items()}

    summary_list = []
    for ticker in all_index_tickers:
        x_val = rs_ratio[ticker].dropna().iloc[-1]
        y_val = rs_momentum[ticker].dropna().iloc[-1]
        
        if x_val >= 100 and y_val >= 100: quad = "LEADING"
        elif x_val >= 100 and y_val < 100: quad = "WEAKENING"
        elif x_val < 100 and y_val < 100: quad = "LAGGING"
        else: quad = "IMPROVING"
            
        summary_list.append({
            "Show": True,
            "SECTOR INDEX": inv_sector_map[ticker],
            "QUADRANT": quad,
            "PERFORMANCE %": round(pct_changes[ticker], 2),
            "TICKER": ticker
        })

    df_master = pd.DataFrame(summary_list)

    # --- STOCKMOJO GRID SPLIT [1, 2.3] ---
    col_left, col_right = st.columns([1, 2.3])

    with col_left:
        st.markdown("### 🗂️ Indices Watchlist")
        
        edited_df = st.data_editor(
            df_master[["Show", "SECTOR INDEX", "QUADRANT", "PERFORMANCE %"]],
            column_config={
                "Show": st.column_config.CheckboxColumn("View", default=True),
                "PERFORMANCE %": st.column_config.NumberColumn(format="%.2f%%")
            },
            hide_index=True,
            use_container_width=True,
            height=580
        )
        
        active_sectors = edited_df[edited_df["Show"] == True]["SECTOR INDEX"].tolist()
        active_tickers = [sector_indices[s] for s in active_sectors]

    with col_right:
        st.markdown("### 📊 Relative Rotation Graph Canvas")
        if len(active_tickers) >= 1:
            all_x, all_y = [], []
            sector_tail_lengths = {}

            for ticker in active_tickers:
                perf_abs = abs(pct_changes[ticker])
                calc_tail = int(np.clip(base_tail_days + int(perf_abs / 2), 3, 15))
                sector_tail_lengths[ticker] = calc_tail
                
                all_x.extend(rs_ratio[ticker].dropna().iloc[-calc_tail:].values)
                all_y.extend(rs_momentum[ticker].dropna().iloc[-calc_tail:].values)

            # Zoom Calculation Matrix
            zoom_offset = zoom_level * 0.5
            min_x, max_x = 100.0 - zoom_offset + center_shift_x, 100.0 + zoom_offset + center_shift_x
            min_y, max_y = 100.0 - zoom_offset + center_shift_y, 100.0 + zoom_offset + center_shift_y

            # Canvas Frame Configuration
            fig, ax = plt.subplots(figsize=(13, 9.5), facecolor='#0e1118')
            ax.set_facecolor('#0e1118')

            # StockMojo Premium Background Shading Colors
            ax.axvspan(100, max_x + 5, ymin=0.5, ymax=1.0, facecolor='#0b1d16', alpha=0.9) # LEADING
            ax.axvspan(100, max_x + 5, ymin=0.0, ymax=0.5, facecolor='#1f1b11', alpha=0.9) # WEAKENING
            ax.axvspan(min_x - 5, 100, ymin=0.0, ymax=0.5, facecolor='#221415', alpha=0.9) # LAGGING
            ax.axvspan(min_x - 5, 100, ymin=0.5, ymax=1.0, facecolor='#0b1826', alpha=0.9) # IMPROVING

            # Clean sleek axis borders
            ax.axhline(100, color='#1e293b', linestyle='-', linewidth=1.5, zorder=3)
            ax.axvline(100, color='#1e293b', linestyle='-', linewidth=1.5, zorder=3)
            ax.grid(True, color='#161f30', linestyle='-', linewidth=0.6, alpha=0.7, zorder=1)

            # Placement text overlays matching exact colors
            ax.text(max_x - (zoom_offset*0.03), max_y - (zoom_offset*0.03), 'Leading', color='#00e676', fontsize=12, fontweight='bold', ha='right', va='top')
            ax.text(max_x - (zoom_offset*0.03), min_y + (zoom_offset*0.03), 'Weakening', color='#ffd700', fontsize=12, fontweight='bold', ha='right', va='bottom')
            ax.text(min_x + (zoom_offset*0.03), min_y + (zoom_offset*0.03), 'Lagging', color='#ff5252', fontsize=12, fontweight='bold', ha='left', va='bottom')
            ax.text(min_x + (zoom_offset*0.03), max_y - (zoom_offset*0.03), 'Improving', color='#00b0ff', fontsize=12, fontweight='bold', ha='left', va='top')

            cmap = plt.colormaps.get_cmap('gist_rainbow')
            colors = [cmap(i) for i in np.linspace(0, 0.9, len(active_tickers))]

            # Interpolation Spline Plotting for Indices
            for idx, ticker in enumerate(active_tickers):
                t_len = sector_tail_lengths[ticker]
                x_trail = rs_ratio[ticker].dropna().iloc[-t_len:].values
                y_trail = rs_momentum[ticker].dropna().iloc[-t_len:].values
                
                sector_color = colors[idx]
                t = np.arange(len(x_trail))
                t_new = np.linspace(0, len(x_trail) - 1, 60)
                
                spl_x = make_interp_spline(t, x_trail, k=3)
                spl_y = make_interp_spline(t, y_trail, k=3)
                
                ax.plot(spl_x(t_new), spl_y(t_new), linestyle='-', linewidth=2.5, color=sector_color, alpha=0.8, zorder=5)
                ax.scatter(x_trail[:-1], y_trail[:-1], color=sector_color, s=15, alpha=0.5, zorder=5)
                ax.scatter(x_trail[-1], y_trail[-1], color=sector_color, s=90, edgecolors='white', linewidth=1.5, zorder=6)
                
                ax.text(x_trail[-1], y_trail[-1] + (zoom_offset*0.02), inv_sector_map[ticker], color='#ffffff', 
                        fontsize=8, fontweight='bold', ha='center', zorder=7)

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.tick_params(colors='#475569', labelsize=9)
            ax.set_xlabel('Trend (RS-Ratio)', color='#64748b', labelpad=10)
            ax.set_ylabel('Momentum (RS-Momentum)', color='#64748b', labelpad=10)
            
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Left panel matrix se kisi bhi Sector Index par tick kijiye, dynamic rotation live load ho jayegi.")
else:
    st.error("Live indexing feed pipeline error.")
