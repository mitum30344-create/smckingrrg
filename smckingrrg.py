import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta

# Page configuration for complete layout scaling
st.set_page_config(page_title="RRG Professional Studio Dashboard", layout="wide")

# --- PREMIUM STOCKMOJO STYLING BLOCK ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1118; color: #e2e8f0; font-family: 'Inter', sans-serif; }
    div.stButton > button:first-child { background-color: #00e676; color: #0e1118; border: none; font-weight: bold; }
    .stSelectbox, .stSlider { color: #ffffff !important; }
    .stExpander { background-color: #161b26 !important; border: 1px solid #232d3f !important; border-radius: 6px !important; margin-bottom: 8px !important;}
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Relative Rotation Graph (RRG) Studio")

# --- 1. SECTOR / STOCK CORE DATA ---
sector_map = {
    "Information Technology (IT)": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS"],
    "Banking & Finance (BFSI)": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "SHRIRAMFIN.NS", "SBILIFE.NS"],
    "Automobile (Auto)": ["BAJAJ-AUTO.NS", "M&M.NS", "MARUTI.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS"],
    "Pharmaceuticals (Pharma)": ["CIPLA.NS", "DRREDDY.NS", "SUNPHARMA.NS"],
    "Consumer Goods / Retail": ["HINDUNILVR.NS", "TRENT.NS", "TITAN.NS", "NESTLEIND.NS"],
    "Energy & Infrastructure": ["RELIANCE.NS", "NTPC.NS", "ADANIPORTS.NS", "GRASIM.NS", "INDIGO.NS"]
}

all_unique_stocks = list(set([stock for sector in sector_map.values() for stock in sector]))

# --- 2. SIDEBAR ENGINE CONTROLS ---
st.sidebar.markdown("### ⚙️ Engine Configurations")
timeframe_option = st.sidebar.selectbox("TIMEFRAME", ["Daily", "1 Hour", "Weekly"])
benchmark_option = st.sidebar.selectbox("BENCHMARK", ["Nifty 500 (^CRSLDX)", "Nifty 50 (^NSEI)", "Nifty Bank (^NSEBANK)"])
base_tail_days = st.sidebar.slider("COUNTS / DAYS (Tail)", min_value=3, max_value=15, value=5)

st.sidebar.markdown("### 🔍 Canvas Zoom Engine")
zoom_level = st.sidebar.slider("Zoom View Scale", min_value=1, max_value=10, value=4)
center_shift_x = st.sidebar.slider("Center Alignment (X)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
center_shift_y = st.sidebar.slider("Center Alignment (Y)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)

ticker_benchmark = benchmark_option.split("(")[-1].replace(")", "")

if timeframe_option == "1 Hour":
    interval, days_back, pct_period = "1h", 30, 7
elif timeframe_option == "Weekly":
    interval, days_back, pct_period = "1wk", 730, 4
else:
    interval, days_back, pct_period = "1d", 500, 5

end_date = datetime.today()
start_date = end_date - timedelta(days=days_back)

with st.spinner("Syncing Live Market Feeds..."):
    data = yf.download(all_unique_stocks + [ticker_benchmark], start=start_date, end=end_date, interval=interval)

close_prices = data['Close'].dropna()
volumes = data['Volume'].loc[close_prices.index]

last_prices = close_prices[all_unique_stocks].iloc[-1]
pct_changes = close_prices[all_unique_stocks].pct_change(periods=pct_period).iloc[-1] * 100

rs_df = pd.DataFrame(index=close_prices.index)
for stock in all_unique_stocks:
    rs_df[stock] = close_prices[stock] / close_prices[ticker_benchmark]

rs_mean = rs_df.rolling(window=14).mean()
rs_std = rs_df.rolling(window=14).std()
rs_ratio = 100 + ((rs_df - rs_mean) / (rs_std + 1e-8)) * 1.2

rs_roc = rs_df.pct_change(periods=5)
vol_mean = volumes[all_unique_stocks].rolling(window=14).mean()
vol_std = volumes[all_unique_stocks].rolling(window=14).std()
norm_volume = (volumes[all_unique_stocks] - vol_mean) / (vol_std + 1e-8)

raw_momentum = rs_roc * np.tanh(norm_volume)
mom_mean = raw_momentum.rolling(window=14).mean()
mom_std = raw_momentum.rolling(window=14).std()
rs_momentum = 100 + ((raw_momentum - mom_mean) / (mom_std + 1e-8)) * 1.2

summary_list = []
for stock in all_unique_stocks:
    x_val = rs_ratio[stock].dropna().iloc[-1]
    y_val = rs_momentum[stock].dropna().iloc[-1]
    
    if x_val >= 100 and y_val >= 100: quad = "LEADING"
    elif x_val >= 100 and y_val < 100: quad = "WEAKENING"
    elif x_val < 100 and y_val < 100: quad = "LAGGING"
    else: quad = "IMPROVING"
        
    summary_list.append({
        "Active": True,
        "SYMBOL": stock.replace('.NS', ''),
        "QUADRANT": quad,
        "PRICE (₹)": round(last_prices[stock], 2),
        "CHANGE %": round(pct_changes[stock], 2),
        "FULL_TICKER": stock
    })

df_master = pd.DataFrame(summary_list)

# --- FIXED GRID PROPORTION RATIO ---
col_left, col_right = st.columns([1, 2.3]) 

active_tickers = []

with col_left:
    st.markdown("### 📁 Sector Folders")
    for sector_name, stock_list in sector_map.items():
        clean_names = [s.replace('.NS', '') for s in stock_list]
        df_sector_subset = df_master[df_master["SYMBOL"].isin(clean_names)].copy()
        
        with st.expander(f"📁 {sector_name}"):
            edited_df = st.data_editor(
                df_sector_subset[["Active", "SYMBOL", "QUADRANT", "CHANGE %"]],
                column_config={
                    "Active": st.column_config.CheckboxColumn("View", default=True),
                    "CHANGE %": st.column_config.NumberColumn(format="%.2f%%")
                },
                hide_index=True,
                use_container_width=True,
                key=f"grid_{sector_name}"
            )
            sub_active = edited_df[edited_df["Active"] == True]["SYMBOL"].tolist()
            active_tickers.extend([s + ".NS" for s in sub_active])

with col_right:
    st.markdown("### 🎛️ RRG Visual Canvas")
    if len(active_tickers) >= 1:
        all_x, all_y = [], []
        stock_tail_lengths = {}

        for stock in active_tickers:
            stock_perf = abs(pct_changes[stock])
            calculated_tail = int(np.clip(base_tail_days + int(stock_perf / 2), 3, 15))
            stock_tail_lengths[stock] = calculated_tail
            
            all_x.extend(rs_ratio[stock].dropna().iloc[-calculated_tail:].values)
            all_y.extend(rs_momentum[stock].dropna().iloc[-calculated_tail:].values)

        zoom_offset = zoom_level * 0.5
        min_x, max_x = 100.0 - zoom_offset + center_shift_x, 100.0 + zoom_offset + center_shift_x
        min_y, max_y = 100.0 - zoom_offset + center_shift_y, 100.0 + zoom_offset + center_shift_y

        fig, ax = plt.subplots(figsize=(13, 9), facecolor='#0e1118')
        ax.set_facecolor('#0e1118')

        # --- EXACT STOCKMOJO UNIFORM HIGH-CONTRAST SHADES ---
        ax.axvspan(100, max_x + 5, ymin=0.5, ymax=1.0, facecolor='#0b1d16', alpha=0.9) # LEADING
        ax.axvspan(100, max_x + 5, ymin=0.0, ymax=0.5, facecolor='#1f1b11', alpha=0.9) # WEAKENING
        ax.axvspan(min_x - 5, 100, ymin=0.0, ymax=0.5, facecolor='#221415', alpha=0.9) # LAGGING
        ax.axvspan(min_x - 5, 100, ymin=0.5, ymax=1.0, facecolor='#0b1826', alpha=0.9) # IMPROVING

        # Matrix axes
        ax.axhline(100, color='#1e293b', linestyle='-', linewidth=1.5, zorder=3)
        ax.axvline(100, color='#1e293b', linestyle='-', linewidth=1.5, zorder=3)
        ax.grid(True, color='#161f30', linestyle='-', linewidth=0.6, alpha=0.7, zorder=1)

        # Dynamic clean labels aligned with StockMojo design protocols
        ax.text(max_x - (zoom_offset*0.03), max_y - (zoom_offset*0.03), 'LEADING', color='#00e676', fontsize=11, fontweight='bold', ha='right', va='top')
        ax.text(max_x - (zoom_offset*0.03), min_y + (zoom_offset*0.03), 'WEAKENING', color='#ffd700', fontsize=11, fontweight='bold', ha='right', va='bottom')
        ax.text(min_x + (zoom_offset*0.03), min_y + (zoom_offset*0.03), 'LAGGING', color='#ff5252', fontsize=11, fontweight='bold', ha='left', va='bottom')
        ax.text(min_x + (zoom_offset*0.03), max_y - (zoom_offset*0.03), 'IMPROVING', color='#00b0ff', fontsize=11, fontweight='bold', ha='left', va='top')

        cmap = plt.colormaps.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(active_tickers))]

        for idx, stock in enumerate(active_tickers):
            t_len = stock_tail_lengths[stock]
            x_trail = rs_ratio[stock].dropna().iloc[-t_len:].values
            y_trail = rs_momentum[stock].dropna().iloc[-t_len:].values
            
            stock_color = colors[idx]
            t = np.arange(len(x_trail))
            t_new = np.linspace(0, len(x_trail) - 1, 60)
            
            spl_x = make_interp_spline(t, x_trail, k=3)
            spl_y = make_interp_spline(t, y_trail, k=3)
            
            ax.plot(spl_x(t_new), spl_y(t_new), linestyle='-', linewidth=2.2, color=stock_color, alpha=0.8, zorder=5)
            ax.scatter(x_trail[:-1], y_trail[:-1], color=stock_color, s=15, alpha=0.5, zorder=5)
            ax.scatter(x_trail[-1], y_trail[-1], color=stock_color, s=85, edgecolors='white', linewidth=1.2, zorder=6)
            
            ax.text(x_trail[-1], y_trail[-1] + (zoom_offset*0.02), stock.replace('.NS',''), color='#ffffff', 
                    fontsize=8, fontweight='bold', ha='center', zorder=7)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.tick_params(colors='#475569', labelsize=9)
        
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Left folders me se stocks par tick kijiye, graph bada aur clean dikhega.")
