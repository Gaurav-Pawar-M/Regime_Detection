import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# CONFIG & STYLING
# ---------------------------------------------------------
# Custom CSS for dark theme and fonts
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #4fc3f7 !important;
    }
    [data-testid="stMetricValue"] {
        color: #4fc3f7;
        white-space: normal !important;
        overflow-wrap: break-word !important;
        font-size: 1.8rem !important;
        word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)

REGIME_COLORS = {
    "Bear": "#ef5350",
    "Bear/Crash": "#ef5350",
    "Sideways": "#78909c",
    "Bull": "#66bb6a",
    "Bullish": "#66bb6a"
}

REGIME_COLORS_RGBA = {
    "Bear": "rgba(239,83,80,0.25)",
    "Bear/Crash": "rgba(239,83,80,0.25)",
    "Sideways": "rgba(120,144,156,0.2)",
    "Bull": "rgba(102,187,106,0.25)",
    "Bullish": "rgba(102,187,106,0.25)"
}

TICKERS = [
    "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "DRREDDY.NS",
    "HDFCBANK.NS", "HINDUNILVR.NS", "ICICIBANK.NS", "INFY.NS", "KOTAKBANK.NS",
    "LT.NS", "MARUTI.NS", "NESTLEIND.NS", "ONGC.NS", "RELIANCE.NS",
    "SUNPHARMA.NS", "TCS.NS", "TATAMOTORS.NS", "WIPRO.NS"
]

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
SRC_DIR = BASE_DIR / "src"

# ---------------------------------------------------------
# DATA LOADING (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_parquet_data():
    log_returns = pd.read_parquet(DATA_DIR / "log_returns.parquet")
    volatility = pd.read_parquet(DATA_DIR / "volatility.parquet")
    price_levels = pd.read_parquet(DATA_DIR / "price_levels.parquet")
    
    events_df = pd.DataFrame()
    if (DATA_DIR / "events_table.parquet").exists():
        events_df = pd.read_parquet(DATA_DIR / "events_table.parquet")
        
    disagreements = pd.DataFrame()
    if (DATA_DIR / "disagreement_table.parquet").exists():
        disagreements = pd.read_parquet(DATA_DIR / "disagreement_table.parquet")
        
    return log_returns, volatility, price_levels, events_df, disagreements

@st.cache_data
def load_nifty_states():
    baseline_nifty = pd.DataFrame()
    nh_nifty = pd.DataFrame()
    if (DATA_DIR / "baseline_hmm_states_nifty.parquet").exists():
        baseline_nifty = pd.read_parquet(DATA_DIR / "baseline_hmm_states_nifty.parquet")
    if (DATA_DIR / "nh_hmm_states_nifty.parquet").exists():
        nh_nifty = pd.read_parquet(DATA_DIR / "nh_hmm_states_nifty.parquet")
    return baseline_nifty, nh_nifty

@st.cache_data
def load_stock_states(symbol):
    if symbol == "^NSEI":
        baseline_nifty, nh_nifty = load_nifty_states()
        df = pd.DataFrame({'date': baseline_nifty['date']}) if not baseline_nifty.empty else pd.DataFrame()
        if not df.empty:
            df['baseline_state'] = baseline_nifty['state']
            df['baseline_label'] = baseline_nifty['state_label']
            df['nh_state'] = nh_nifty['state']
            df['nh_label'] = nh_nifty['state_label']
        return df
        
    path = DATA_DIR / "per_stock_states" / f"{symbol}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

@st.cache_resource
def load_hmm_metrics():
    # Note: Transition matrices in hmm_metrics.json are known-broken (random Dirichlet noise)
    # due to NIFTY lacking event indicators during EM fit. We only load it for LL stats.
    metrics = {}
    if (DATA_DIR / "hmm_metrics.json").exists():
        with open(DATA_DIR / "hmm_metrics.json", "r") as f:
            metrics = json.load(f)
    return metrics

@st.cache_resource
def load_hmm_metrics_avg():
    metrics = {}
    if (DATA_DIR / "hmm_metrics_per_stock_avg.json").exists():
        with open(DATA_DIR / "hmm_metrics_per_stock_avg.json", "r") as f:
            metrics = json.load(f)
    return metrics

@st.cache_resource
def load_xgb():
    path = SRC_DIR / "xgboost_classifier.pkl"
    if path.exists():
        return joblib.load(path)
    return None

@st.cache_data
def load_xgb_features():
    path = DATA_DIR / "xgb_features_full.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()

with st.spinner("Loading NSE Regime Engine..."):
    log_returns, volatility, price_levels, events_df, disagreements = load_parquet_data()
    baseline_nifty, nh_nifty = load_nifty_states()
    hmm_metrics = load_hmm_metrics()
    hmm_metrics_avg = load_hmm_metrics_avg()
    xgb_model = load_xgb()
    xgb_features = load_xgb_features()

# ---------------------------------------------------------
# SECTION 1: SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("NSE Regime Engine")

all_options = TICKERS + ["NIFTY50 Index (^NSEI)"]
selected_option = st.sidebar.selectbox("Select Stock/Index", all_options)
selected_ticker = "^NSEI" if "NIFTY50" in selected_option else selected_option

min_date = price_levels.index.min()
max_date = price_levels.index.max()
default_start = max_date - pd.Timedelta(days=730) if pd.notnull(max_date) else min_date

date_range = st.sidebar.date_input("Date Range", value=(default_start, max_date), min_value=min_date, max_value=max_date)

st.sidebar.divider()

st.sidebar.markdown("### Model Statistics")
if not disagreements.empty and "baseline_ll" in disagreements.columns and "nh_ll" in disagreements.columns:
    ll_df = disagreements[['symbol', 'baseline_ll', 'nh_ll']].drop_duplicates()
    if not ll_df.empty:
        ll_df['ll_imp'] = ((ll_df['nh_ll'] - ll_df['baseline_ll']) / ll_df['baseline_ll'].abs()) * 100
        avg_imp = ll_df['ll_imp'].mean()
        num_beat = (ll_df['nh_ll'] > ll_df['baseline_ll']).sum()
        total_stocks = len(ll_df)
        st.sidebar.markdown(f"**NH-HMM outperforms baseline HMM in log-likelihood for {num_beat} of {total_stocks} stocks**")
        st.sidebar.text(f"(average improvement: {avg_imp:.2f}%)")

if "^NSEI" in log_returns.columns:
    st.sidebar.text(f"Training Days (T): {len(log_returns[['^NSEI']].dropna())}")

A_normal = hmm_metrics_avg.get("A_normal_avg")
A_event = hmm_metrics_avg.get("A_event_avg")
if A_normal and A_event:
    A_normal = np.array(A_normal)
    A_event = np.array(A_event)
    norm_diff = hmm_metrics_avg.get("frobenius_norm", np.linalg.norm(A_event - A_normal, ord='fro'))
    st.sidebar.text(f"17-stock avg (Fro): {norm_diff:.4f}")


    
st.sidebar.text(f"Total Disagreements: {len(disagreements)}")

verified_manual_df = pd.DataFrame()
if os.path.exists("data/disagreement_table_verified_MANUAL.csv"):
    try:
        verified_manual_df = pd.read_csv("data/disagreement_table_verified_MANUAL.csv")
    except:
        pass

if not verified_manual_df.empty and 'verdict' in verified_manual_df.columns:
    justified_count = (verified_manual_df['verdict'] == 'Justified').sum()
    not_justified_count = (verified_manual_df['verdict'] == 'Not Justified').sum()
    uncertain_count = (verified_manual_df['verdict'] == 'Uncertain').sum()
    denom = justified_count + not_justified_count
    justified_rate = justified_count / denom if denom > 0 else 0
    total_disagreements = len(disagreements)
    
    unique_symbols = verified_manual_df['symbol'].nunique()
    unique_events = verified_manual_df['event_type'].nunique()
    
    stats_text1 = f"The NH-HMM was validated against {len(verified_manual_df)} hand-verified cases across {unique_symbols} stocks and {unique_events} event types with a {justified_rate*100:.1f}% justified rate (excluding Uncertain)."
    stats_text2 = f"**NH-HMM justified in {justified_count} of {denom} hand-verified cases with a clear beat/miss ({uncertain_count} additional cases marked Uncertain; {total_disagreements} total disagreement cases exist in the full dataset)**"
else:
    stats_text1 = "Validation metrics are pending manual verification."
    stats_text2 = "**Verification pending**"

with st.sidebar.expander("Methodology", expanded=False):
    st.markdown(f"""
    This engine applies a Non-Homogeneous HMM that switches between two
    transition matrices: A_normal for regular trading days and A_event for
    days within ±5 trading days of a scheduled corporate event (Results,
    Board Meeting, AGM). {stats_text1}
    """)

if len(date_range) != 2:
    st.warning("Please select a complete date range.")
    st.stop()
    
start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

stock_states = load_stock_states(selected_ticker)
if stock_states.empty:
    st.warning(f"No regime states found for {selected_ticker}. Fallback to NIFTY50 if necessary.")
else:
    label_counts = stock_states["nh_label"].value_counts()
    if label_counts.get("Bear/Crash", 0) > 0.7 * len(stock_states):
        st.warning("Note: This stock shows predominantly Bear regime — may reflect 2022-2024 sector weakness, not a bug.")

# ---------------------------------------------------------
# SECTION 2: HEADER METRICS
# ---------------------------------------------------------
if not stock_states.empty:
    latest_state = stock_states.iloc[-1]
    
    current_baseline_label = latest_state.get('baseline_label', 'N/A')
    current_nh_label = latest_state.get('nh_label', 'N/A')
    
    # Event logic
    last_event_str = "None"
    days_since_event = "N/A"
    
    if not events_df.empty:
        sym_events = events_df[events_df['symbol'] == selected_ticker].copy()
        if not sym_events.empty:
            sym_events['event_date'] = pd.to_datetime(sym_events['event_date'])
            past_events = sym_events[sym_events['event_date'] <= max_date]
            if not past_events.empty:
                last_event = past_events.iloc[-1]
                last_event_str = f"{last_event['event_date'].strftime('%Y-%m-%d')} ({last_event['event_type']})"
                days_since_event = (max_date - last_event['event_date']).days

    # XGBoost Signal - Real XGBoost probability using all model features
    feature_path = SRC_DIR / "xgb_features.json"
    if feature_path.exists():
        with open(feature_path, "r") as f:
            FEATURES = json.load(f)
    else:
        FEATURES = []
    
    signal = "HOLD"
    xgb_prob = None
    
    if xgb_model is not None and not xgb_features.empty:
        sym_feats = xgb_features[xgb_features['symbol'] == selected_ticker]
        if not sym_feats.empty:
            # Ensure all features exist in the dataframe before slicing
            avail_feats = [f for f in FEATURES if f in sym_feats.columns]
            latest = sym_feats.iloc[[-1]][avail_feats]
            xgb_prob = float(xgb_model.predict_proba(latest)[0, 1])
            if "Bear" in current_nh_label:
                signal = "AVOID"
            elif "Bull" in current_nh_label and xgb_prob > 0.53:
                signal = "BUY"
            else:
                signal = "HOLD"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Current Regime (Baseline)", str(current_baseline_label))
    col2.metric("Current Regime (NH-HMM)", str(current_nh_label))
    col3.metric("Last Corporate Event", last_event_str)
    col4.metric("Days Since Event", str(days_since_event))
    
    # Show as metric with probability delta
    col5.metric(
        "XGBoost Signal",
        signal,
        delta=f"P(Up 5d): {xgb_prob:.3f}" if xgb_prob is not None else "Model not loaded",
        delta_color="off"
    )
    
    if xgb_prob is not None:
        sub_col1, sub_col2, sub_col3 = st.columns(3)
        edge_pct = abs(xgb_prob - 0.5) * 2 * 100
        sub_col1.metric("Model Edge", f"{edge_pct:.1f}%")
        sub_col2.metric("Regime Confidence", "N/A")
        
        if 'E_t' in latest.columns and 'days_to_event' in latest.columns:
            in_window = latest['E_t'].iloc[0] == 1
            d2e = latest['days_to_event'].iloc[0]
            # Convert to int in case it's float
            window_str = f"YES ({int(d2e)} days)" if in_window else "NO"
        else:
            window_str = "N/A"
        sub_col3.metric("Event Window", window_str)

st.markdown("---")

# ---------------------------------------------------------
# SECTION 3: REGIME HISTORY CHART
# ---------------------------------------------------------
st.markdown(f"### Regime History: {selected_ticker}")

if selected_ticker in price_levels.columns and not stock_states.empty:
    mask = (price_levels.index >= start_dt) & (price_levels.index <= end_dt)
    df_prices = price_levels.loc[mask, selected_ticker].dropna()
    
    mask_states = (stock_states['date'] >= start_dt) & (stock_states['date'] <= end_dt)
    df_states = stock_states.loc[mask_states].copy()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_prices.index, y=df_prices.values,
        mode='lines', name='Price', line=dict(color='#c9d1d9')
    ))
    
    if not df_states.empty:
        df_states['shift_label'] = df_states['nh_label'].shift(1)
        df_states['regime_change'] = df_states['nh_label'] != df_states['shift_label']
        change_indices = df_states[df_states['regime_change']].index
        
        start_idx = df_states.index[0]
        for i in range(1, len(change_indices)):
            end_idx = change_indices[i]
            regime = df_states.loc[start_idx, 'nh_label']
            color = REGIME_COLORS_RGBA.get(regime, "rgba(120,144,156,0.2)")
            
            fig.add_vrect(
                x0=df_states.loc[start_idx, 'date'], 
                x1=df_states.loc[end_idx, 'date'],
                fillcolor=color, opacity=1, layer="below", line_width=0
            )
            start_idx = end_idx
            
        regime = df_states.loc[start_idx, 'nh_label']
        color = REGIME_COLORS_RGBA.get(regime, "rgba(120,144,156,0.2)")
        fig.add_vrect(
            x0=df_states.loc[start_idx, 'date'], 
            x1=df_states.iloc[-1]['date'],
            fillcolor=color, opacity=1, layer="below", line_width=0
        )
        
    if not events_df.empty:
        sym_events = events_df[events_df['symbol'] == selected_ticker].copy()
        sym_events['event_date'] = pd.to_datetime(sym_events['event_date'])
        sym_events = sym_events[(sym_events['event_date'] >= start_dt) & (sym_events['event_date'] <= end_dt)]
        
        event_colors = {"Results": "red", "Board Meeting": "blue", "AGM": "green"}
        for _, row in sym_events.iterrows():
            ev_color = event_colors.get(row['event_type'], "white")
            fig.add_vline(x=row['event_date'], line_dash="dash", line_color=ev_color, opacity=0.8, name=row['event_type'])

    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Price (INR)",
        xaxis_title="Date",
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# SECTION 4: TRANSITION MATRIX HEATMAPS
# ---------------------------------------------------------
st.markdown("### Regime Transition Probabilities")
st.markdown("Non-Homogeneous HMMs adapt their transition matrix during corporate event windows. The true aggregate across all 17 validated stocks reveals that transitions into the `Bear/Crash` state actually **decrease slightly** during event windows, contrary to what the NIFTY-only baseline suggested.")

if A_normal is not None and A_event is not None:
    matrix_options = ["Average (17 stocks)"] + [t for t in TICKERS if t != "TATAMOTORS.NS"]
    selected_matrix_view = st.selectbox("Select Matrix View", matrix_options, index=0)
    
    if selected_matrix_view == "Average (17 stocks)":
        st.caption("This heatmap shows the 17-stock average. Individual stocks vary substantially -- standard deviation exceeds the mean difference in every transition cell, meaning no single stock closely matches this average pattern. See the per-stock breakdown for actual stock-level dynamics.")
        disp_A_normal, disp_A_event = A_normal, A_event
    else:
        matrix_path = DATA_DIR / f"matrices_{selected_matrix_view}.json"
        if matrix_path.exists():
            with open(matrix_path, "r") as f:
                d = json.load(f)
                disp_A_normal, disp_A_event = np.array(d["A_normal"]), np.array(d["A_event"])
            matrix_title_label = selected_matrix_view
        else:
            st.warning(f"No matrices found for {selected_matrix_view}.")
            disp_A_normal, disp_A_event = A_normal, A_event
            matrix_title_label = f"{selected_matrix_view} (FALLBACK: Average)"

    col_t1, col_t2 = st.columns(2)
    
    fig_norm = px.imshow(disp_A_normal, 
                         labels=dict(x="To State", y="From State", color="Prob"),
                         x=["Bear", "Neutral", "Bull"], y=["Bear", "Neutral", "Bull"],
                         text_auto='.2f', color_continuous_scale='Blues')
    fig_norm.update_layout(title=f"A_normal ({matrix_title_label})", template="plotly_dark")
    col_t1.plotly_chart(fig_norm, use_container_width=True)
    
    fig_event = px.imshow(disp_A_event, 
                          labels=dict(x="To State", y="From State", color="Prob"),
                          x=["Bear", "Neutral", "Bull"], y=["Bear", "Neutral", "Bull"],
                          text_auto='.2f', color_continuous_scale='Reds')
    fig_event.update_layout(title=f"A_event ({matrix_title_label})", template="plotly_dark")
    col_t2.plotly_chart(fig_event, use_container_width=True)
    
    norm_diff = np.linalg.norm(disp_A_event - disp_A_normal, ord='fro')
    st.markdown(f"**Frobenius Norm Difference (A_event - A_normal):** `{norm_diff:.4f}`")

# ---------------------------------------------------------
# SECTION 5: BACKTESTING PANEL
# ---------------------------------------------------------
st.markdown(f"### Strategy Backtesting ({selected_ticker})")

if not stock_states.empty and selected_ticker in log_returns.columns:
    bt_df = pd.DataFrame({'date': log_returns.index, 'log_return': log_returns[selected_ticker].fillna(0)})
    bt_df = pd.merge(bt_df, stock_states[['date', 'nh_label', 'baseline_label']], on='date', how='left')
    
    bt_df['nh_label'] = bt_df['nh_label'].ffill()
    bt_df['baseline_label'] = bt_df['baseline_label'].ffill()
    
    bt_df['bh_return'] = bt_df['log_return']
    bt_df['nh_strat_return'] = np.where(bt_df['nh_label'].isin(["Bear", "Bear/Crash"]), 0.0, bt_df['log_return'])
    bt_df['base_strat_return'] = np.where(bt_df['baseline_label'].isin(["Bear", "Bear/Crash"]), 0.0, bt_df['log_return'])
    
    bt_df = bt_df[(bt_df['date'] >= start_dt) & (bt_df['date'] <= end_dt)]
    
    bt_df['bh_cum'] = 100 * np.exp(bt_df['bh_return'].cumsum())
    bt_df['nh_cum'] = 100 * np.exp(bt_df['nh_strat_return'].cumsum())
    bt_df['base_cum'] = 100 * np.exp(bt_df['base_strat_return'].cumsum())
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=bt_df['date'], y=bt_df['bh_cum'], name="Buy & Hold", line=dict(color="rgba(255,255,255,0.3)", width=2)))
    fig_bt.add_trace(go.Scatter(x=bt_df['date'], y=bt_df['base_cum'], name="Baseline Strategy", line=dict(color="#ffb300", dash="dash", width=3)))
    fig_bt.add_trace(go.Scatter(x=bt_df['date'], y=bt_df['nh_cum'], name="NH-HMM Strategy", line=dict(color="#4fc3f7", width=1.5)))
    
    fig_bt.update_layout(template="plotly_dark", yaxis_title="Portfolio Value (Index=100)", xaxis_title="Date", margin=dict(l=40, r=40, t=40, b=40))
    st.plotly_chart(fig_bt, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True})
    
    def calc_metrics(returns):
        tot_ret = (np.exp(returns.sum()) - 1) * 100
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        cum_ret = returns.cumsum()
        drawdown = cum_ret - cum_ret.cummax()
        max_dd = (1 - np.exp(drawdown.min())) * 100
        return sharpe, max_dd, tot_ret
        
    bh_s, bh_dd, bh_tr = calc_metrics(bt_df['bh_return'])
    nh_s, nh_dd, nh_tr = calc_metrics(bt_df['nh_strat_return'])
    bs_s, bs_dd, bs_tr = calc_metrics(bt_df['base_strat_return'])
    
    mcol1, mcol2, mcol3 = st.columns(3)
    
    with mcol1:
        st.markdown("**Buy & Hold**")
        st.text(f"Sharpe Ratio (Ann): {bh_s:.2f}")
        st.text(f"Max Drawdown: {bh_dd:.2f}%")
        st.text(f"Total Return: {bh_tr:.2f}%")
        
    with mcol2:
        st.markdown("**NH-HMM Strategy**")
        st.text(f"Sharpe Ratio (Ann): {nh_s:.2f}")
        st.text(f"Max Drawdown: {nh_dd:.2f}%")
        st.text(f"Total Return: {nh_tr:.2f}%")
        
    with mcol3:
        st.markdown("**Baseline Strategy**")
        st.text(f"Sharpe Ratio (Ann): {bs_s:.2f}")
        st.text(f"Max Drawdown: {bs_dd:.2f}%")
        st.text(f"Total Return: {bs_tr:.2f}%")

# ---------------------------------------------------------
# SECTION 6: DISAGREEMENT TABLE
# ---------------------------------------------------------
st.markdown("### Model Disagreements on Event Dates")
st.markdown(stats_text2)

if not disagreements.empty:
    disp_filter = st.selectbox("Filter Disagreements by Stock", ["All"] + TICKERS)
    disp_df = disagreements.copy()
    if disp_filter != "All":
        disp_df = disp_df[disp_df['symbol'] == disp_filter]
        
    if disp_df.empty:
        st.info(f"No model disagreements found for {disp_filter}.")
    else:
        show_cols = ['event_date', 'symbol', 'event_type', 'baseline_label', 'nh_label', 'A_event_norm']
        # Intersect with available columns safely
        show_cols = [c for c in show_cols if c in disp_df.columns]
        disp_df = disp_df[show_cols].copy()
        
        def add_badge(label):
            if label in ['Bull', 'Bullish', 'Sideways']: return f"🟢 {label}"
            elif label in ['Bear', 'Bear/Crash', 'Bearish']: return f"🔴 {label}"
            return label
            
        if 'nh_label' in disp_df.columns:
            disp_df['nh_label'] = disp_df['nh_label'].apply(add_badge)
        
        def color_code(row):
            val = str(row.get('nh_label', ''))
            if '🟢' in val:
                return ['background-color: rgba(102,187,106,0.2)'] * len(row)
            elif '🔴' in val:
                return ['background-color: rgba(239,83,80,0.25)'] * len(row)
            return [''] * len(row)
            
        st.dataframe(disp_df.style.apply(color_code, axis=1), use_container_width=True)


# ---------------------------------------------------------
# SECTION 7: SHAP PANEL
# ---------------------------------------------------------
st.markdown("### Signal Explanation (SHAP)")
st.info("Signal requires BOTH NH-HMM Bull regime AND XGBoost P(Up)>53% for BUY. NH-HMM Bear overrides to AVOID.")

if xgb_model is not None and not xgb_features.empty:
    import shap
    sym_feats = xgb_features[xgb_features['symbol'] == selected_ticker]
    
    if not sym_feats.empty:
        avail_feats = [f for f in FEATURES if f in sym_feats.columns]
        latest = sym_feats.iloc[[-1]][avail_feats]
        
        explainer = shap.TreeExplainer(xgb_model)
        sv = explainer.shap_values(latest)
        
        # Handle different shape outputs from shap_values
        sv_arr = sv[1] if isinstance(sv, list) else sv
        if len(sv_arr.shape) == 3:
            sv_arr = sv_arr[0, :, 1]
        else:
            sv_arr = sv_arr[0]
            
        shap_df = pd.DataFrame({
            'Feature': avail_feats,
            'SHAP': sv_arr
        }).sort_values('SHAP')
        
        colors = ['#ef5350' if v < 0 else '#66bb6a' for v in shap_df['SHAP']]
        
        fig_shap = go.Figure(go.Bar(
            x=shap_df['SHAP'], y=shap_df['Feature'],
            orientation='h', marker_color=colors
        ))
        
        fig_shap.update_layout(
            template="plotly_dark",
            title=f"Feature Impact for Signal: {signal}",
            xaxis_title="SHAP Value",
            margin=dict(l=40, r=40, t=40, b=40)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
    else:
        st.info("No XGBoost features available for this ticker.")
else:
    st.info("Run notebook 11_shap_feature_attribution.ipynb first to enable this panel.")
