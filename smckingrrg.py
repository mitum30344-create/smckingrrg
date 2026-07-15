import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta

# Page config to force full fluid wide-screen like Strike Money
st.set_page_config(page_title="RRG Professional Dashboard", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #151924; color: white; }
    div.stButton > button:first-child { background-color: #26a69a; color: white; border: none; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Relative Rotation Graph (RRG) - Premium Studio")

# --- 1. SECTOR / STOCK CORE DATA ---
sector_map = {
    "All Sectors": [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
        "BHARTIARTL.NS", "SBIN.NS", "HCLTECH.NS", "TECHM.NS", "TATASTEEL.NS",
        "HINDALCO.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS", "COALINDIA.NS", "HINDUNILVR.NS",
        "CIPLA.NS", "M&M.NS", "SHRIRAMFIN.NS", "DRREDDY.NS", "WIPRO.NS",
        "JSWSTEEL.NS", "TRENT.NS", "BAJFINANCE.NS", "ADANIPORTS.NS", "INDIGO.NS", 
        "GRASIM.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS"
    ],
    "Information Technology (IT)": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "TECHM.NS", "WIPRO.NS"],
    "Banking & Finance (BFSI)": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "BAJFINANCE.NS", "SHRIRAMFIN.NS", "SBILIFE.NS"],
    "Automobile (Auto)": ["BAJAJ-AUTO.NS", "M&M.NS", "MARUTI.NS"],
    "Metals & Mining": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "COALINDIA.NS"],
    "Pharmaceuticals (Pharma)": ["CIPLA.NS", "DRREDDY.NS", "SUNPHARMA.NS"],
    "Consumer Goods / Retail": ["HINDUNILVR.NS", "TRENT.NS", "TITAN.NS", "NESTLEIND.NS"],
    "Energy & Infrastructure": ["RELIANCE.NS", "NTPC.NS", "ADANIPORTS.NS", "GRASIM.NS", "INDIGO.NS"]
}

# --- 2. SIDEBAR PREMIUM INPUT CONTROL ENGINE ---
st.sidebar.markdown("### ⚙️ Engine Configurations")
timeframe_option = st.sidebar.selectbox("TIMEFRAME", ["Daily", "1 Hour", "Weekly"])
benchmark_option = st.sidebar.selectbox("BENCHMARK", ["Nifty 500 (^CRSLDX)", "Nifty 50 (^NSEI)", "Nifty Bank (^NSEBANK)"])
selected_sector = st.sidebar.selectbox("SECTOR FILTER", list(sector_map.keys()))
base_tail_days = st.sidebar.slider("COUNTS / DAYS (Tail)", min_value=3, max_value=15, value=5)

ticker_benchmark = benchmark_option.split("(")[-1].replace(")", "")

if timeframe_option == "1 Hour":
    interval, days_back, pct_period = "1h", 30, 7
elif timeframe_option == "Weekly":
    interval, days_back, pct_period = "1wk", 730, 4
else:
    interval, days_back, pct_period = "1d", 500, 5

end_date = datetime.today()
start_date = end_date - timedelta(days=days_back)
sector_stocks = sector_map[selected_sector]

with st.spinner("Syncing Live Market Feeds..."):
    data = yf.download(sector_stocks + [ticker_benchmark], start=start_date, end=end_date, interval=interval)

if 'Close' in data and not data['Close'].empty:
    close_prices = data['Close'].dropna()
    volumes = data['Volume'].loc[close_prices.index]
    
    # Live Last Price & % Change calculation
    last_prices = close_prices[sector_stocks].iloc[-1]
    pct_changes = close_prices[sector_stocks].pct_change(periods=pct_period).iloc[-1] * 100

    # RRG Vector Engineering Matrix
    rs_df = pd.DataFrame(index=close_prices.index)
    for stock in sector_stocks:
        rs_df[stock] = close_prices[stock] / close_prices[ticker_benchmark]

    rs_mean = rs_df.rolling(window=14).mean()
    rs_std = rs_df.rolling(window=14).std()
    rs_ratio = 100 + ((rs_df - rs_mean) / (rs_std + 1e-8)) * 1.2

    rs_roc = rs_df.pct_change(periods=5)
    vol_mean = volumes[sector_stocks].rolling(window=14).mean()
    vol_std = volumes[sector_stocks].rolling(window=14).std()
    norm_volume = (volumes[sector_stocks] - vol_mean) / (vol_std + 1e-8)

    raw_momentum = rs_roc * np.tanh(norm_volume)
    mom_mean = raw_momentum.rolling(window=14).mean()
    mom_std = raw_momentum.rolling(window=14).std()
    rs_momentum = 100 + ((raw_momentum - mom_mean) / (mom_std + 1e-8)) * 1.2

    # Parse and structural loop generation
    summary_list = []
    for stock in sector_stocks:
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
            "raw_x": x_ratio_val := rs_ratio[stock].dropna(),
            "raw_y": y_momentum_val := rs_momentum[stock].dropna()
        })

    df_controls = pd.DataFrame(summary_list)

    # --- STRIKE MONEY UI GRID SPLIT ---
    # Left Column: Interactive Watchlist Table Matrix | Right Column: Clean RRG Chart Frame [1]
    col_left, col_right = st.columns([2, 3]) 

    with col_left:
        st.markdown("### 📋 Symbols Matrix")
        # Streamlit Data Editor acts as clickable table lists with checkboxes [1]
        edited_df = st.data_editor(
            df_controls[["Active", "SYMBOL", "QUADRANT", "PRICE (₹)", "CHANGE %"]],
            column_config={
                "Active": st.column_config.CheckboxColumn("View", default=True),
                "CHANGE %": st.column_config.NumberColumn(format="%.2f%%"),
                "PRICE (₹)": st.column_config.NumberColumn(format="₹%.2f")
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Filter stocks live based on checkbox table selections
        active_symbols = edited_df[edited_df["Active"] == True]["SYMBOL"].tolist()
        active_tickers = [s + ".NS" for s in active_symbols]

    with col_right:
        if len(active_tickers) >= 1:
            all_x, all_y = [], []
            stock_tail_lengths = {}

            for stock in active_tickers:
                stock_perf = abs(pct_changes[stock])
                calculated_tail = int(np.clip(base_tail_days + int(stock_perf / 2), 3, 15))
                stock_tail_lengths[stock] = calculated_tail
                
                all_x.extend(rs_ratio[stock].dropna().iloc[-calculated_tail:].values)
                all_y.extend(rs_momentum[stock].dropna().iloc[-calculated_tail:].values)

            min_x, max_x = min(all_x) - 0.4, max(all_x) + 0.4
            min_y, max_y = min(all_y) - 0.4, max(all_y) + 0.4

            # --- ULTRA CLEAN GRAPH FRAME ---
            fig, ax = plt.subplots(figsize=(11, 8.5), facecolor='#151924')
            ax.set_facecolor('#151924')

            # Soft transperant uniform quadrants
            ax.axvspan(100, max_x + 5, ymin=0.5, ymax=1.0, facecolor='#162620', alpha=0.9) # Leading
            ax.axvspan(100, max_x + 5, ymin=0.0, ymax=0.5, facecolor='#2b241a', alpha=0.9) # Weakening
            ax.axvspan(min_x - 5, 100, ymin=0.0, ymax=0.5, facecolor='#2a1a1c', alpha=0.9) # Lagging
            ax.axvspan(min_x - 5, 100, ymin=0.5, ymax=1.0, facecolor='#162032', alpha=0.9) # Improving

            # Thin clean axis borders
            ax.axhline(100, color='#2c3240', linestyle='-', linewidth=1.5, zorder=3)
            ax.axvline(100, color='#2c3240', linestyle='-', linewidth=1.5, zorder=3)
            ax.grid(True, color='#202430', linestyle='-', linewidth=0.6, alpha=0.7, zorder=1)

            # Clean outer corner quadrant labels matching Strike Money [1]
            ax.text(max_x, max_y, 'LEADING', color='#26a69a', fontsize=11, fontweight='bold', ha='right', va='top')
            ax.text(max_x, min_y, 'WEAKENING', color='#ffb300', fontsize=11, fontweight='bold', ha='right', va='bottom')
            ax.text(min_x, min_y, 'LAGGING', color='#ef5350', fontsize=11, fontweight='bold', ha='left', va='bottom')
            ax.text(min_x, max_y, 'IMPROVING', color='#29b6f6', fontsize=11, fontweight='bold', ha='left', va='top')

            cmap = plt.colormaps.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, len(active_tickers))]

            # Premium Spline Curves Plotting Engine
            for idx, stock in enumerate(active_tickers):
                t_len = stock_tail_lengths[stock]
                x_trail = rs_ratio[stock].dropna().iloc[-t_len:].values
                y_trail = rs_momentum[stock].dropna().iloc[-t_len:].values
                
                stock_color = colors[idx]
                t = np.arange(len(x_trail))
                t_new = np.linspace(0, len(x_trail) - 1, 60) # High density subpoints
                
                spl_x = make_interp_spline(t, x_trail, k=3)
                spl_y = make_interp_spline(t, y_trail, k=3)
                
                # Plot dynamic fluid line curves
                ax.plot(spl_x(t_new), spl_y(t_new), linestyle='-', linewidth=2.0, color=stock_color, alpha=0.8, zorder=5)
                ax.scatter(x_trail[:-1], y_trail[:-1], color=stock_color, s=15, alpha=0.5, zorder=5)
                ax.scatter(x_trail[-1], y_trail[-1], color=stock_color, s=80, edgecolors='white', linewidth=1.2, zorder=6)
                
                # Small elegant floating stock name tag text
                ax.text(x_trail[-1], y_trail[-1] + 0.05, stock.replace('.NS',''), color='#ffffff', 
                        fontsize=8, fontweight='bold', ha='center', zorder=7)

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.tick_params(colors='#616d82', labelsize=9)
            
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Please select checkboxes from the left panel table to load live trailing vectors.")
else:
    st.error("Data processing downtime. Check back during market sessions.")
