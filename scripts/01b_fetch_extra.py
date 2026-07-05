"""
01b_fetch_extra.py
==================
Fetches and engineers advanced technical indicators for all 19 NSE stocks
plus NIFTY50 index. Saves to ../data/extra_features.parquet

Run independently: python scripts/01b_fetch_extra.py

Does NOT overwrite any existing parquet files.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
START    = "2019-01-01"
END      = "2026-07-04"  # Changed END to 2026-07-04 so it fetches latest data

TICKERS = [
    "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "DRREDDY.NS",
    "HDFCBANK.NS",   "HINDUNILVR.NS", "ICICIBANK.NS", "INFY.NS",
    "KOTAKBANK.NS",  "LT.NS", "MARUTI.NS", "NESTLEIND.NS",
    "ONGC.NS",       "RELIANCE.NS", "SUNPHARMA.NS", "TCS.NS",
    "TATAMOTORS.NS", "WIPRO.NS", "^NSEI"
]

# ── Indicator Functions ──────────────────────────────────────────────

def compute_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(close, fast=12, slow=26, signal=9):
    ema_fast   = close.ewm(span=fast, adjust=False).mean()
    ema_slow   = close.ewm(span=slow, adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(close, period=20, num_std=2):
    ma    = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    pct_b = (close - lower) / (upper - lower + 1e-9)  # %B: 0=at lower, 1=at upper
    bandwidth = (upper - lower) / (ma + 1e-9)
    return upper, lower, pct_b, bandwidth

def compute_atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(d_period).mean()
    return k, d

def compute_features_for_ticker(sym, hist, nifty_returns=None):
    """
    Given a yfinance history DataFrame, compute all technical features.
    Returns a DataFrame with one row per trading day.
    """
    close  = hist['Close']
    high   = hist['High']
    low    = hist['Low']
    volume = hist['Volume']

    df = pd.DataFrame(index=hist.index)

    # ── 1. Volume Features ───────────────────────────────
    df['volume_ratio'] = volume / volume.rolling(20).mean()
    df['volume_ratio'] = df['volume_ratio'].clip(0, 10)  # cap outliers

    # ── 2. Intraday Range ────────────────────────────────
    df['hl_range'] = (high - low) / close  # normalised intraday range

    # ── 3. RSI ───────────────────────────────────────────
    df['rsi_14'] = compute_rsi(close, 14)
    df['rsi_7']  = compute_rsi(close, 7)   # faster RSI for event sensitivity

    # ── 4. MACD ──────────────────────────────────────────
    macd, signal_line, histogram = compute_macd(close)
    df['macd']          = macd
    df['macd_signal']   = signal_line
    df['macd_hist']     = histogram
    df['macd_cross']    = (macd > signal_line).astype(int)  # 1=bullish cross

    # ── 5. Bollinger Bands ───────────────────────────────
    bb_upper, bb_lower, pct_b, bw = compute_bollinger(close)
    df['bb_pct_b']      = pct_b      # where price sits within bands (0-1)
    df['bb_bandwidth']  = bw         # band width = volatility proxy
    df['bb_squeeze']    = (bw < bw.rolling(20).mean()).astype(int)  # 1=squeeze

    # ── 6. ATR (normalised by price) ─────────────────────
    atr = compute_atr(high, low, close)
    df['atr_pct'] = atr / close      # ATR as % of price — comparable across stocks

    # ── 7. Stochastic Oscillator ─────────────────────────
    stoch_k, stoch_d = compute_stochastic(high, low, close)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['stoch_oversold']  = (stoch_k < 20).astype(int)
    df['stoch_overbought']= (stoch_k > 80).astype(int)

    # ── 8. Moving Average Features ───────────────────────
    ma_20  = close.rolling(20).mean()
    ma_50  = close.rolling(50).mean()
    ma_200 = close.rolling(200).mean()

    df['dist_ma20']  = (close - ma_20)  / ma_20    # % distance from 20d MA
    df['dist_ma50']  = (close - ma_50)  / ma_50
    df['dist_ma200'] = (close - ma_200) / ma_200
    df['ma_trend']   = (ma_20 > ma_50).astype(int)  # 1=short MA above long MA

    # ── 9. Momentum ──────────────────────────────────────
    log_ret = np.log(close / close.shift(1))
    df['momentum_5d']  = log_ret.rolling(5).sum().shift(1)   # shift: no lookahead
    df['momentum_20d'] = log_ret.rolling(20).sum().shift(1)
    df['momentum_60d'] = log_ret.rolling(60).sum().shift(1)

    # ── 10. Volatility Regime ────────────────────────────
    df['realised_vol_5d']  = log_ret.rolling(5).std()  * np.sqrt(252)
    df['realised_vol_20d'] = log_ret.rolling(20).std() * np.sqrt(252)
    df['vol_regime']       = (df['realised_vol_5d'] > df['realised_vol_20d']).astype(int)

    # ── 11. Market Regime (NIFTY50 context) ──────────────
    if nifty_returns is not None and sym != '^NSEI':
        nifty_aligned = nifty_returns.reindex(df.index).ffill()
        df['nifty_momentum_5d']  = nifty_aligned.rolling(5).sum().shift(1)
        df['nifty_momentum_20d'] = nifty_aligned.rolling(20).sum().shift(1)
        df['nifty_above_ma50']   = (
            nifty_aligned.cumsum().rolling(50).mean() <
            nifty_aligned.cumsum()
        ).astype(int)
    else:
        df['nifty_momentum_5d']  = np.nan
        df['nifty_momentum_20d'] = np.nan
        df['nifty_above_ma50']   = np.nan

    # ── 12. Price Patterns ───────────────────────────────
    # Gap up/down from previous close
    df['overnight_gap'] = (close - close.shift(1)) / close.shift(1)
    df['gap_up']   = (df['overnight_gap'] >  0.01).astype(int)
    df['gap_down'] = (df['overnight_gap'] < -0.01).astype(int)

    # ── 13. Lag Features ─────────────────────────────────
    df['lag_return_1'] = log_ret.shift(1)
    df['lag_return_2'] = log_ret.shift(2)
    df['lag_return_5'] = log_ret.shift(5)
    df['lag_vol_1']    = df['realised_vol_5d'].shift(1)

    return df


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("01b_fetch_extra.py — Advanced Feature Engineering")
    print("=" * 60)

    # First fetch NIFTY50 for market regime features
    print("\n[0/19] Fetching NIFTY50 index for market regime context...")
    try:
        nifty_hist = yf.download("^NSEI", start=START, end=END,
                                  auto_adjust=True, progress=False)
        if isinstance(nifty_hist.columns, pd.MultiIndex):
            nifty_hist.columns = ['_'.join(c).strip() for c in nifty_hist.columns]
            close_col = [c for c in nifty_hist.columns if 'Close' in c][0]
            nifty_close = nifty_hist[close_col]
        else:
            nifty_close = nifty_hist['Close']
        nifty_returns = np.log(nifty_close / nifty_close.shift(1))
        print(f"    NIFTY50: {len(nifty_close)} days loaded")
    except Exception as e:
        print(f"    WARNING: Could not load NIFTY50: {e}")
        nifty_returns = None

    # Fetch and compute features for each stock
    all_features = {}
    failed = []

    for i, sym in enumerate(TICKERS):
        print(f"\n[{i+1}/{len(TICKERS)}] Processing {sym}...")
        try:
            hist = yf.download(sym, start=START, end=END,
                               auto_adjust=True, progress=False)

            if hist.empty:
                print(f"    WARNING: No data for {sym}")
                failed.append(sym)
                continue

            # Flatten MultiIndex if present
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = [c[0] for c in hist.columns]

            df = compute_features_for_ticker(sym, hist, nifty_returns)
            all_features[sym] = df

            non_null = df.notna().sum().sum()
            total    = df.shape[0] * df.shape[1]
            print(f"    OK: {len(df)} days, {len(df.columns)} features "
                  f"({100*non_null/total:.1f}% non-null)")

        except Exception as e:
            print(f"    ERROR for {sym}: {e}")
            failed.append(sym)

    if not all_features:
        print("\nERROR: No features computed. Check internet connection.")
        return

    # Combine into one DataFrame with MultiIndex columns (sym, feature)
    combined = pd.concat(all_features, axis=1)
    combined.index.name = 'date'

    # Save
    out_path = DATA_DIR / "extra_features.parquet"
    combined.to_parquet(out_path)

    print("\n" + "=" * 60)
    print(f"SAVED: {out_path}")
    print(f"Shape: {combined.shape}")
    print(f"Date range: {combined.index.min().date()} -> {combined.index.max().date()}")
    print(f"Stocks processed: {len(all_features)}/{len(TICKERS)}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Features per stock: {combined.shape[1] // len(all_features)}")
    print("=" * 60)

    # Quick sanity check
    print("\nSample feature values (RELIANCE.NS, last 3 rows):")
    if 'RELIANCE.NS' in all_features:
        print(all_features['RELIANCE.NS'].tail(3)[
            ['rsi_14', 'macd_hist', 'bb_pct_b', 'atr_pct', 'stoch_k',
             'dist_ma50', 'momentum_20d']
        ].round(4).to_string())


if __name__ == "__main__":
    main()
