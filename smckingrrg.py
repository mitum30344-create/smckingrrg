import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta

# Website Page Configuration
st.set_page_config(page_title="Smart Money RRG Dashboard", layout="wide")

st.title("📊 Volume-Weighted Relative Rotation Graph (RRG)")
st.sidebar.header("📊 Settings Dashboard")

# 1. Sidebar Inputs
benchmark_option = st.sidebar.selectbox(
    "Select Benchmark Index",
    ["Nifty 500 (^CRSLDX)", "Nifty 50 (^NSEI)", "Nifty Bank (^NSEBANK)"]
)
ticker_benchmark = benchmark_option.split("(")[-1].replace(")", "")

tail_days = st.sidebar.slider("Select Trail Length (Days)", min_value=3, max_value=15, value=5)

# Date calculations (Automatically fetches up to current live market date)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=500)).strftime('%Y-%m-%d')

# 2. Complete Stock List
tickers_stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS", 
    "BHARTIARTL.NS", "SBIN.NS", "HCLTECH.NS", "TECHM.NS", "TATASTEEL.NS",
    "HINDALCO.NS", "BAJAJ-AUTO.NS", "SBILIFE.NS", "COALINDIA.NS", "HINDUNILVR.NS",
    "CIPLA.NS", "M&M.NS", "SHRIRAMFIN.NS", "DRREDDY.NS", "WIPRO.NS",
    "JSWSTEEL.NS", "TRENT.NS", "BAJFINANCE.NS", "ADANIPORTS.NS", "INDIGO.NS", 
    "GRASIM.NS", "TITAN.NS", "SUNPHARMA.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS"
]

# Multi-select dropdown to add/remove stocks live on website
selected_stocks = st.sidebar.multiselect("Add / Remove Stocks", tickers_stocks, default=tickers_stocks[:15])

if st.sidebar.button("Run Live Analysis") or selected_stocks:
    with st.spinner("Fetching Live Market Data from Yahoo Finance..."):
        # Download Data
        data = yf.download(selected_stocks + [ticker_benchmark], start=start_date, end=end_date)
        close_prices = data['Close'].dropna()
        volumes = data['Volume'].loc[close_prices.index]

        # RRG Mathematical Logic
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

        # Setup Plotting Canvas
        fig, ax = plt.subplots(figsize=(14, 10), facecolor='#151924')
        ax.set_facecolor('#151924')

        all_x, all_y = [], []
        for stock in selected_stocks:
            all_x.extend(rs_ratio[stock].iloc[-tail_days:].values)
            all_y.extend(rs_momentum[stock].iloc[-tail_days:].values)

        min_x, max_x = min(all_x) - 0.5, max(all_x) + 0.5
        min_y, max_y = min(all_y) - 0.5, max(all_y) + 0.5

        # Background Layout Tint
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
        colors = [cmap(i) for i in np.linspace(0, 1, len(selected_stocks))]

        # Curve Rendering
        for idx, stock in enumerate(selected_stocks):
            x_trail = rs_ratio[stock].iloc[-tail_days:].values
            y_trail = rs_momentum[stock].iloc[-tail_days:].values
            
            if not np.isnan(x_trail).any() and not np.isnan(y_trail).any():
                stock_color = colors[idx]
                t = np.arange(len(x_trail))
                t_new = np.linspace(0, len(x_trail) - 1, 50)
                
                spl_x = make_interp_spline(t, x_trail, k=3)
                spl_y = make_interp_spline(t, y_trail, k=3)
                
                ax.plot(spl_x(t_new), spl_y(t_new), linestyle='-', linewidth=2.2, color=stock_color, alpha=0.8, zorder=5)
                ax.scatter(x_trail[:-1], y_trail[:-1], color=stock_color, s=20, alpha=0.4, zorder=5)
                ax.scatter(x_trail[-1], y_trail[-1], color=stock_color, s=90, edgecolors='white', linewidth=1.5, zorder=6)
                ax.annotate(stock.replace('.NS', ''), (x_trail[-1], y_trail[-1]), textcoords="offset points", xytext=(0,8), 
                            ha='center', color='#ffffff', fontsize=9, fontweight='bold', zorder=7)

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.tick_params(colors='#8a94a6', labelsize=10)
        
        # Display the live chart on webpage
        st.pyplot(fig)
