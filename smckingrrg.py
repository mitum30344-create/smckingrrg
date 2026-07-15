import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from datetime import datetime, timedelta

# Page configuration for complete layout optimization
st.set_page_config(page_title="RRG Professional Studio Dashboard", layout="wide")

# Custom Premium Dark Theme CSS
st.markdown("""
    <style>
    .stApp { background-color: #0e1118; color: #cbd5e1; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
    .stSelectbox, .stSlider { color: #ffffff !important; }
    .stExpander { background-color: #161b26 !important; border: 1px solid #232d3f !important; border-radius: 6px !important; margin-bottom: 6px !important; }
    div[data-testid="stDataFrame"] { background-color: #161b26 !important; border-radius: 6px; }
    button[data-baseweb="tab"] { font-size: 15px !important; font-weight: bold !important; color: #8a94a6 !important; }
    button[aria-selected="true"] { color: #00e676 !important; border-bottom-color: #00e676 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Professional Multi-Timeframe Volume-Weighted RRG")

# --- 1. CORE SECTOR TO STOCKS INDEXING MAP ---
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
ticker_benchmark = "^CRSLDX" # Nifty 500

# --- 2. SIDEBAR ENGINE CONTROLS ---
st.sidebar.markdown("### ⚙️ Engine Settings")
timeframe_option = st.sidebar.selectbox("Timeframe", ["Daily", "Weekly"])
base_tail_days = st.sidebar.slider("Tail Length (Days)", min_value=3, max_value=15, value=5)

if timeframe_option == "Weekly":
    interval, days_back, pct_period = "1wk", 730, 4
else:
    interval, days_back, pct_period = "1d", 500, 5

end_date = datetime.today()
start_date = end_date - timedelta(days=days_back)

with st.spinner("Downloading Global Live Market Feed Matrix..."):
    raw_data = yf.download(all_unique_stocks + [ticker_benchmark], start=start_date, end=end_date, interval=interval)

if not raw_data.empty and 'Close' in raw_data:
    close_prices = raw_data['Close'].dropna(subset=[ticker_benchmark])
    volumes = raw_data['Volume'].loc[close_prices.index]

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

    master_summary = []
    for stock in all_unique_stocks:
        x_val = rs_ratio[stock].dropna().iloc[-1]
        y_val = rs_momentum[stock].dropna().iloc[-1]
        
        if x_val >= 100 and y_val >= 100: quad = "LEADING"
        elif x_val >= 100 and y_val < 100: quad = "WEAKENING"
        elif x_val < 100 and y_val < 100: quad = "LAGGING"
        else: quad = "IMPROVING"
            
        master_summary.append({
            "Active": True,
            "SYMBOL": stock.replace('.NS', ''),
            "QUADRANT": quad,
            "PERF %": round(pct_changes[stock], 2),
            "TICKER": stock
        })

    df_master = pd.DataFrame(master_summary)

    # --- SECTOR FOLDERS IN SIDEBAR PANEL ---
    st.sidebar.markdown("### 🗂️ Sector Folders")
    active_tickers = []
    
    for sector_name, stock_list in sector_map.items():
        clean_names = [s.replace('.NS', '') for s in stock_list]
        df_subset = df_master[df_master["SYMBOL"].isin(clean_names)].copy()
        
        with st.sidebar.expander(f"📁 {sector_name}"):
            edited_df = st.data_editor(
                df_subset[["Active", "SYMBOL"]],
                column_config={"Active": st.column_config.CheckboxColumn("View", default=True)},
                hide_index=True,
                use_container_width=True,
                key=f"side_exp_{sector_name}"
            )
            sub_active = edited_df[edited_df["Active"] == True]["SYMBOL"].tolist()
            active_tickers.extend([s + ".NS" for s in sub_active])

    # --- Main Frame Dashboard Workspace ---
    if len(active_tickers) >= 1:
        all_x, all_y = [], []
        stock_tail_lengths = {}

        for stock in active_tickers:
            perf_abs = abs(pct_changes[stock])
            calc_tail = int(np.clip(base_tail_days + int(perf_abs / 2), 3, 15))
            stock_tail_lengths[stock] = calc_tail
            
            all_x.extend(rs_ratio[stock].dropna().iloc[-calc_tail:].values)
            all_y.extend(rs_momentum[stock].dropna().iloc[-calc_tail:].values)

        min_x, max_x = min(all_x) - 0.4, max(all_x) + 0.4
        min_y, max_y = min(all_y) - 0.4, max(all_y) + 0.4

        # Matplotlib High-Fidelity Canvas
        fig, ax = plt.subplots(figsize=(15, 8.5), facecolor='#0e1118')
        ax.set_facecolor('#0e1118')

        ax.axvspan(100, max_x + 5, ymin=0.5, ymax=1.0, facecolor='#0b1d16', alpha=0.9) 
        ax.axvspan(100, max_x + 5, ymin=0.0, ymax=0.5, facecolor='#1f1b11', alpha=0.9) 
        ax.axvspan(min_x - 5, 100, ymin=0.0, ymax=0.5, facecolor='#221415', alpha=0.9) 
        ax.axvspan(min_x - 5, 100, ymin=0.5, ymax=1.0, facecolor='#0b1826', alpha=0.9) 

        ax.axhline(100, color='#1e293b', linestyle='-', linewidth=1.5, zorder=3)
        ax.axvline(100, color='#1e293b', linestyle='-', linewidth=1.5, zorder=3)
        ax.grid(True, color='#161f30', linestyle='-', linewidth=0.6, alpha=0.7, zorder=1)

        ax.text(max_x - 0.05, max_y - 0.05, 'LEADING', color='#00e676', fontsize=12, fontweight='bold', ha='right', va='top')
        ax.text(max_x - 0.05, min_y + 0.05, 'WEAKENING', color='#ffd700', fontsize=12, fontweight='bold', ha='right', va='bottom')
        ax.text(min_x + 0.05, min_y + 0.05, 'LAGGING', color='#ff5252', fontsize=12, fontweight='bold', ha='left', va='bottom')
        ax.text(min_x + 0.05, max_y - 0.05, 'IMPROVING', color='#00b0ff', fontsize=12, fontweight='bold', ha='left', va='top')

        cmap = plt.colormaps.get_cmap('gist_rainbow')
        colors = [cmap(i) for i in np.linspace(0, 0.9, len(active_tickers))]

        table_rows = []

        for idx, stock in enumerate(active_tickers):
            t_len = stock_tail_lengths[stock]
            x_trail = rs_ratio[stock].dropna().iloc[-t_len:].values
            y_trail = rs_momentum[stock].dropna().iloc[-t_len:].values
            
            stock_color = colors[idx]
            t = np.arange(len(x_trail))
            t_new = np.linspace(0, len(x_trail) - 1, 60)
            
            spl_x = make_interp_spline(t, x_trail, k=3)
            spl_y = make_interp_spline(t, y_trail, k=3)
            
            ax.plot(spl_x(t_new), spl_y(t_new), linestyle='-', linewidth=2.5, color=stock_color, alpha=0.8, zorder=5)
            ax.scatter(x_trail[:-1], y_trail[:-1], color=stock_color, s=15, alpha=0.5, zorder=5)
            ax.scatter(x_trail[-1], y_trail[-1], color=stock_color, s=90, edgecolors='white', linewidth=1.5, zorder=6)
            
            ax.text(x_trail[-1], y_trail[-1] + 0.015, stock.replace('.NS',''), color='#ffffff', 
                    fontsize=8, fontweight='bold', ha='center', zorder=7)

            x_last, y_last = x_trail[-1], y_trail[-1]
            if x_last >= 100 and y_last >= 100: q_state = "LEADING"
            elif x_last >= 100 and y_last < 100: q_state = "WEAKENING"
            elif x_last < 100 and y_last < 100: q_state = "LAGGING"
            else: q_state = "IMPROVING"

            table_rows.append({
                "SYMBOL": stock.replace('.NS', ''),
                "QUADRANT": q_state,
                "RS-RATIO": round(x_last, 2),
                "RS-MOMENTUM": round(y_last, 2),
                "RETURN %": round(pct_changes[stock], 2)
            })

        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.tick_params(colors='#475569', labelsize=9)
        st.pyplot(fig, use_container_width=True)

        # --- FIX: SPACES AND TABS PROPERLY ALIGNED FOR ALL TABS BLOCKS ---
        st.markdown("### 📋 Quadrant Allocation Matrix")
        df_summary = pd.DataFrame(table_rows)

        tab_lead, tab_imp, tab_weak, tab_lag = st.tabs(["🟩 Leading", "🟦 Improving", "🟨 Weakening", "🟥 Lagging"])

        with tab_lead:
            df_lead = df_summary[df_summary["QUADRANT"] == "LEADING"][["SYMBOL", "RS-RATIO", "RS-MOMENTUM", "RETURN %"]]
            st.dataframe(df_lead.sort_values(by="RETURN %", ascending=False), hide_index=True, use_container_width=True)

        with tab_imp:
            df_imp = df_summary[df_summary["QUADRANT"] == "IMPROVING"][["SYMBOL", "RS-RATIO", "RS-MOMENTUM", "RETURN %"]]
