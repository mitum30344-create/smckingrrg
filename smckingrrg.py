import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta

# Website Page Configuration (Wide-screen forced)
st.set_page_config(page_title="Professional Wide RRG Dashboard", layout="wide")

st.title("📊 Dynamic Multi-Timeframe Volume-Weighted RRG")
st.sidebar.header("📊 Advanced Controls")

# --- 1. SECTOR MAP DEFINITION ---
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

# --- 2. MULTI-TIMEFRAME DROPDOWN CONFIGURATION ---
timeframe_option = st.sidebar.selectbox(
    "Select Timeframe",
    ["Daily (Swing)", "1 Hour (Intraday)", "Weekly (Positional)"]
)

if timeframe_option == "1 Hour (Intraday)":
    interval = "1h"
    days_back = 30  
    pct_period = 7  
elif timeframe_option == "Weekly (Positional)":
    interval = "1wk"
    days_back = 730 
    pct_period = 4  
else:
    interval = "1d"
    days_back = 500 
    pct_period = 5  

benchmark_option = st.sidebar.selectbox(
    "Select Benchmark Index",
    ["Nifty 500 (^CRSLDX)", "Nifty 50 (^NSEI)", "Nifty Bank (^NSEBANK)"]
)
ticker_benchmark = benchmark_option.split("(")[-1].replace(")", "")

selected_sector = st.sidebar.selectbox("Select Market Sector", list(sector_map.keys()))
base_tail_days = st.sidebar.slider("Base Trail Length (Points)", min_value=4, max_value=15, value=6)

end_date = datetime.today()
start_date = end_date - timedelta(days=days_back)

sector_stocks = sector_map[selected_sector]
selected_stocks = st.sidebar.multiselect("Add / Remove Stocks", sector_stocks, default=sector_stocks)

if (st.sidebar.button("Calculate Dynamic Rotation") or selected_stocks) and len(selected_stocks) >= 2:
    with st.spinner(f"Processing Multi-Timeframe Data ({interval}) for {selected_sector}..."):
        
        data = yf.download(selected_stocks + [ticker_benchmark], start=start_date, end=end_date, interval=interval)
        
        if 'Close' in data and not data['Close'].empty:
            close_prices = data['Close'].dropna()
            volumes = data['Volume'].loc[close_prices.index]

            pct_changes = close_prices[selected_stocks].pct_change(periods=pct_period).iloc[-1] * 100

            rs_df = pd.DataFrame(index=close_prices.index)
            for stock in selected_stocks:
                rs_df[stock] = close_prices[stock] / close_prices[ticker_benchmark]

            rs_mean = rs_df.rolling(window=14).mean()
            rs_std = rs_df.rolling(window=14).std()
            rs_ratio = 100 + ((rs_df - rs_mean) / (rs_std + 1e-8)) * 1.2

            rs_roc = rs_df.pct_change(periods=5)
            vol_mean = volumes[selected_stocks].rolling(window=14).mean()
            vol_std = volumes[selected_stocks].rolling(window=14).std()
            norm_volume = (volumes[selected_stocks] - vol_mean) / (vol_std + 1e-8)

            raw_momentum = rs_roc * np.tanh(norm_volume)
            mom_mean = raw_momentum.rolling(window=14).mean()
            mom_std = raw_momentum.rolling(window=14).std()
            rs_momentum = 100 + ((raw_momentum - mom_mean) / (mom_std + 1e-8)) * 1.2

            all_x, all_y = [], []
            valid_stocks = []
            stock_tail_lengths = {}

            for stock in selected_stocks:
                stock_perf = abs(pct_changes[stock])
                calculated_tail = int(np.clip(base_tail_days + int(stock_perf / 2), 3, 15))
                stock_tail_lengths[stock] = calculated_tail
                
                x_vals = rs_ratio[stock].dropna().iloc[-calculated_tail:].values
                y_vals = rs_momentum[stock].dropna().iloc[-calculated_tail:].values
                if len(x_vals) == calculated_tail and len(y_vals) == calculated_tail:
                    all_x.extend(x_vals)
                    all_y.extend(y_vals)
                    valid_stocks.append(stock)

            if all_x and all_y:
                min_x, max_x = min(all_x) - 0.5, max(all_x) + 0.5
                min_y, max_y = min(all_y) - 0.5, max(all_y) + 0.5
            else:
                min_x, max_x = 98.0, 102.0
                min_y, max_y = 98.0, 102.0

            # --- BIG CHANNELS FORMAT (COLUMNS HATA DIYE HAIN PAR EXTRA WIDTH DE DI HAI) ---
            fig, ax = plt.subplots(figsize=(15, 8.5), facecolor='#151924') # Graph width increased to 15
            ax.set_facecolor('#151924')

            ax.axvspan(100, max_x + 2, ymin=0.5, ymax=1.0, facecolor='#1b2a24', alpha=0.9)
            ax.axvspan(100, max_x + 2, ymin=0.0, ymax=0.5, facecolor='#2c271e', alpha=0.9)
            ax.axvspan(min_x - 2, 100, ymin=0.0, ymax=0.5, facecolor='#2d1f21', alpha=0.9)
            ax.axvspan(min_x - 2, 100, ymin=0.5, ymax=1.0, facecolor='#1b2436', alpha=0.9)

            ax.axhline(100, color='#3a4152', linestyle='-', linewidth=2.0, zorder=3)
            ax.axvline(100, color='#3a4152', linestyle='-', linewidth=2.0, zorder=3)
            ax.grid(True, color='#262c3c', linestyle='-', linewidth=0.8, alpha=0.8, zorder=1)

            ax.text(max_x - 0.2, max_y - 0.2, 'Leading', color='#26a69a', fontsize=16, fontweight='bold', ha='right', va='top')
            ax.text(max_x - 0.2, min_y + 0.2, 'Weakening', color='#ffb300', fontsize=16, fontweight='bold', ha='right', va='bottom')
            ax.text(min_x + 0.2, min_y + 0.2, 'Lagging', color='#ef5350', fontsize=16, fontweight='bold', ha='left', va='bottom')
            ax.text(min_x + 0.2, max_y - 0.2, 'Improving', color='#29b6f6', fontsize=16, fontweight='bold', ha='left', va='top')

            cmap = plt.colormaps.get_cmap('rainbow')
            colors = [cmap(i) for i in np.linspace(0, 1, len(valid_stocks))]

            summary_data = []

            for idx, stock in enumerate(valid_stocks):
                t_len = stock_tail_lengths[stock]
                x_trail = rs_ratio[stock].dropna().iloc[-t_len:].values
                y_trail = rs_momentum[stock].dropna().iloc[-t_len:].values
                
                stock_color = colors[idx]
                t = np.arange(len(x_trail))
                t_new = np.linspace(0, len(x_trail) - 1, 50)
                
                spl_x = make_interp_spline(t, x_trail, k=3)
                spl_y = make_interp_spline(t, y_trail, k=3)
                
                ax.plot(spl_x(t_new), spl_y(t_new), linestyle='-', linewidth=2.2, color=stock_color, alpha=0.8, zorder=5)
                ax.scatter(x_trail[:-1], y_trail[:-1], color=stock_color, s=15, alpha=0.4, zorder=5)
                ax.scatter(x_trail[-1], y_trail[-1], color=stock_color, s=90, edgecolors='white', linewidth=1.5, zorder=6)
                
                stock_return = pct_changes[stock]
                sign = "+" if stock_return > 0 else ""
                label_text = f"{stock.replace('.NS','')}\n({sign}{stock_return:.1f}%)"
                
                ax.annotate(label_text, (x_trail[-1], y_trail[-1]), textcoords="offset points", xytext=(0,10), 
                            ha='center', color='#ffffff', fontsize=8, fontweight='bold', zorder=7,
                            bbox=dict(boxstyle="round,pad=0.1", fc='#151924', alpha=0.7, edgecolor='none'))

                latest_x = round(x_trail[-1], 2)
                latest_y = round(y_trail[-1], 2)
                
                if latest_x >= 100 and latest_y >= 100: quadrant = "🟩 Leading"
                elif latest_x >= 100 and latest_y < 100: quadrant = "🟨 Weakening"
                elif latest_x < 100 and latest_y < 100: quadrant = "🟥 Lagging"
                else: quadrant = "🟦 Improving"
                
                summary_data.append({
                    "Stock Symbol": stock.replace('.NS', ''),
                    "Return %": round(stock_return, 2),
                    "Trail Points (Tail)": t_len,
                    "RS-Ratio (Strength)": latest_x,
                    "RS-Momentum (Momentum)": latest_y,
                    "Current State": quadrant
                })

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.tick_params(colors='#8a94a6', labelsize=10)
            
            # Pure wide-scale visualization rendering
            st.pyplot(fig, use_container_width=True)

            # --- DYNAMIC HORIZONTAL TOP TRACKER GRID (CHART KE NICHE) ---
            st.markdown("### 🏆 Live Market Pulse (Top Gainers)")
            df_temp = pd.DataFrame(summary_data).sort_values(by="Return %", ascending=False)
            
            # Creating dynamic grid columns horizontally
            g_cols = st.columns(min(len(df_temp), 5))
            for i, (idx_row, row) in enumerate(df_temp.head(5).iterrows()):
                with g_cols[i]:
                    st.metric(label=row["Stock Symbol"], value=f"{row['Return %']}%", delta=row["Current State"])

