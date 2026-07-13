import pandas as pd
import requests
import time
import os
import shutil
import numpy as np
from datetime import datetime
from bse import BSE

STOCKS = [
    "ASIANPAINT.NS", "AXISBANK.NS", "BAJAJ-AUTO.NS", "DRREDDY.NS", "HDFCBANK.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INFY.NS", "KOTAKBANK.NS", "LT.NS",
    "MARUTI.NS", "NESTLEIND.NS", "ONGC.NS", "RELIANCE.NS", "SUNPHARMA.NS",
    "TCS.NS", "TATAMOTORS.NS", "WIPRO.NS"
]

BSE_MAPPING = {
    "ASIANPAINT.NS": "500820", "AXISBANK.NS": "532215", "BAJAJ-AUTO.NS": "532977",
    "DRREDDY.NS": "500124", "HDFCBANK.NS": "500180", "HINDUNILVR.NS": "500696",
    "ICICIBANK.NS": "532174", "INFY.NS": "500209", "KOTAKBANK.NS": "500247",
    "LT.NS": "500510", "MARUTI.NS": "532500", "NESTLEIND.NS": "500790",
    "ONGC.NS": "500312", "RELIANCE.NS": "500325", "SUNPHARMA.NS": "524715",
    "TCS.NS": "532540", "TATAMOTORS.NS": "500570", "WIPRO.NS": "507685"
}

START_DATE = "2019-01-01"
END_DATE = "2026-07-04"

REPORTING_LAG = {
    3:  (15, 45),   # Q4/FY results (period ending Mar) -> ~mid-Apr to mid-May
    6:  (15, 35),   # Q1 (period ending Jun) -> ~mid-Jul to early-Aug
    9:  (15, 45),   # Q2 (period ending Sep) -> ~mid-Oct to mid-Nov
    12: (15, 35),   # Q3 (period ending Dec) -> ~mid-Jan to early-Feb
}


def robust_request(func, *args, **kwargs):
    """Executes a function with retries and backoff."""
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == 2:
                raise e
            time.sleep(5)


def fetch_bse_events(symbol, start_dt, end_dt):
    """Case 1: Primary source via BSE API."""
    scripcode = BSE_MAPPING.get(symbol)
    if not scripcode:
        return []
    
    events = []
    
    def get_bse_page(page):
        with BSE(download_folder='./bse_cache') as bse_client:
            return bse_client.announcements(
                scripcode=scripcode, 
                from_date=start_dt, 
                to_date=end_dt,
                page_no=page
            )

    page = 1
    while True:
        try:
            res = robust_request(get_bse_page, page)
            table = res.get('Table', []) if isinstance(res, dict) else []
            if not table:
                break
                
            for item in table:
                text = (str(item.get('HEADLINE', '')) + " " + 
                        str(item.get('CATEGORYNAME', '')) + " " + 
                        str(item.get('NEWSSUB', ''))).lower()
                
                ev_type = None
                if 'result' in text:
                    ev_type = 'Results'
                elif 'board meeting' in text:
                    ev_type = 'Board Meeting'
                elif 'agm' in text or 'annual general meeting' in text:
                    ev_type = 'AGM'
                
                if ev_type:
                    date_str = item.get('NEWS_DT') or item.get('DT_TM')
                    if date_str:
                        dt = pd.to_datetime(date_str).normalize()
                        events.append({
                            'symbol': symbol,
                            'event_date': dt,
                            'event_type': ev_type,
                            'date_source': 'BSE_API',
                            'is_estimated': False
                        })
            
            # BSE page size is typically 50. If fewer are returned, it's the last page.
            if len(table) < 50:
                break
                
            page += 1
            time.sleep(2) # polite delay
            
        except Exception as e:
            print(f"[{symbol}] BSE API fetch failed on page {page}: {e}")
            break
            
    return events


def fetch_nse_events(symbol, start_dt, end_dt):
    """Case 2: Secondary source via NSE API."""
    symbol_no_ns = symbol.replace('.NS', '')
    url = "https://www.nseindia.com/api/corporate-announcements"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://www.nseindia.com"}
    
    session = requests.Session()
    
    def get_nse_chunk(s_date, e_date):
        def _call():
            session.get("https://www.nseindia.com", headers=headers, timeout=10)
            time.sleep(1)
            params = {
                'index': 'equities',
                'symbol': symbol_no_ns,
                'from_date': s_date.strftime('%d-%m-%Y'),
                'to_date': e_date.strftime('%d-%m-%Y')
            }
            resp = session.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        return robust_request(_call)
        
    events = []
    current_start = start_dt
    
    while current_start <= end_dt:
        current_end = min(current_start + pd.DateOffset(months=3), end_dt)
        try:
            data = get_nse_chunk(current_start, current_end)
            if data and isinstance(data, list):
                for item in data:
                    text = (str(item.get('desc', '')) + " " + 
                            str(item.get('attchmntText', '')) + " " + 
                            str(item.get('subject', ''))).lower()
                            
                    ev_type = None
                    if 'result' in text:
                        ev_type = 'Results'
                    elif 'board meeting' in text:
                        ev_type = 'Board Meeting'
                    elif 'agm' in text or 'annual general meeting' in text:
                        ev_type = 'AGM'
                    
                    if ev_type:
                        date_str = item.get('an_dt') or item.get('sort_date')
                        if date_str:
                            dt = pd.to_datetime(date_str).normalize()
                            events.append({
                                'symbol': symbol,
                                'event_date': dt,
                                'event_type': ev_type,
                                'date_source': 'NSE_API',
                                'is_estimated': False
                            })
                            
            time.sleep(2) # polite delay
            
        except Exception as e:
            print(f"[{symbol}] NSE API fetch failed for chunk {current_start.date()} to {current_end.date()}: {e}")
        
        current_start = current_end + pd.Timedelta(days=1)
        
    return events


def get_estimated_results(symbol, start_dt, end_dt):
    """Case 3: Last-resort estimate using reporting lags."""
    events = []
    start_year = start_dt.year
    end_year = end_dt.year
    
    for year in range(start_year, end_year + 1):
        quarter_ends = [(3, 31), (6, 30), (9, 30), (12, 31)]
        for m, d in quarter_ends:
            q_end = pd.Timestamp(year, m, d)
            if start_dt <= q_end <= end_dt:
                min_lag, max_lag = REPORTING_LAG[m]
                mid_lag = (min_lag + max_lag) // 2
                est_dt = q_end + pd.Timedelta(days=mid_lag)
                events.append({
                    'symbol': symbol,
                    'event_date': est_dt,
                    'event_type': 'Results',
                    'date_source': 'ESTIMATED_LAG',
                    'is_estimated': True
                })
    return events


def merge_events(df1, df2, symbol):
    if df1.empty and df2.empty:
        return pd.DataFrame()
    df = pd.concat([df1, df2], ignore_index=True)
    if df.empty:
        return df
        
    df = df.sort_values('event_date')
    final_rows = []
    
    for ev_type in df['event_type'].unique():
        sub_df = df[df['event_type'] == ev_type].copy()
        
        # Cluster dates that fall within 45 days of each other (same quarterly event)
        sub_df['cluster'] = (sub_df['event_date'].diff().dt.days > 45).cumsum()
        
        for _, group in sub_df.groupby('cluster'):
            min_date_row = group.loc[group['event_date'].idxmin()]
            
            if len(group) > 1:
                max_date = group['event_date'].max()
                min_date = group['event_date'].min()
                # 3 calendar days covers typical 2 trading days disagreement
                if (max_date - min_date).days > 3:
                    sources = group['date_source'].unique()
                    print(f"[{symbol}] {ev_type} disagreement > 2 days: {min_date.date()} vs {max_date.date()} (Sources: {list(sources)}). Keeping {min_date.date()}.")
            
            final_rows.append(min_date_row)
            
    return pd.DataFrame(final_rows)


def main():
    print("Starting real events extraction pipeline...")
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    
    all_final_events = []
    
    for symbol in STOCKS:
        print(f"\nProcessing {symbol}...")
        df_bse = pd.DataFrame(fetch_bse_events(symbol, start_dt, end_dt))
        df_nse = pd.DataFrame(fetch_nse_events(symbol, start_dt, end_dt))
        df_est = pd.DataFrame(get_estimated_results(symbol, start_dt, end_dt))
        
        merged_real = merge_events(df_bse, df_nse, symbol)
        
        final_all = [merged_real] if not merged_real.empty else []
        
        if not df_est.empty:
            for _, est_row in df_est.iterrows():
                # Check if there is a real 'Results' event within 60 days of this estimate
                if not merged_real.empty:
                    real_results = merged_real[merged_real['event_type'] == 'Results']
                    if not real_results.empty:
                        min_dist = (real_results['event_date'] - est_row['event_date']).abs().min().days
                        if min_dist <= 60:
                            continue # Has real event
                # Add fallback
                final_all.append(pd.DataFrame([est_row]))
                
        if final_all:
            res_df = pd.concat(final_all, ignore_index=True)
            all_final_events.append(res_df)
            counts = res_df.groupby('date_source').size().to_dict()
            print(f"[{symbol}] Saved {len(res_df)} events. Sources: {counts}")
        else:
            print(f"[{symbol}] WARNING: Zero events saved.")

    if all_final_events:
        final_df = pd.concat(all_final_events, ignore_index=True)
        final_df = final_df.sort_values(by=['symbol', 'event_date'])
        
        # Cast to datetime64[ms]
        final_df['event_date'] = final_df['event_date'].astype('datetime64[ms]')
        
        out_path = os.path.join('data', 'events_table_real.parquet')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        if os.path.exists(out_path):
            backup_path = os.path.join('data', 'events_table_real_OLD.parquet')
            shutil.copy(out_path, backup_path)
            print(f"\nBacked up old events to {backup_path}")
            
        final_df.to_parquet(out_path, index=False)
        
        # Validation checks
        quarter_end_dates = {(3,31), (6,30), (9,30), (12,31)}
        results_rows = final_df[final_df['event_type'] == 'Results']
        on_quarter_end = results_rows['event_date'].apply(lambda d: (d.month, d.day) in quarter_end_dates)
        pct_suspicious = on_quarter_end.mean() * 100
        
        print(f"\nResults events falling exactly on quarter-end date: {pct_suspicious:.1f}%")
        if pct_suspicious > 5:
            print("WARNING: High % of Results dates land exactly on quarter-end — this is the signature of a proxy-date bug. Investigate before trusting this data.")
            
        print("\nSUMMARY TABLE BY SOURCE:")
        summary = final_df.groupby(['symbol', 'event_type', 'date_source']).size().unstack(fill_value=0)
        print(summary)
        
        print(f"\nTotal rows extracted: {len(final_df)}")
        print(f"Saved cleanly to {out_path}")
        
    else:
        print("\nFATAL ERROR: No events extracted for any stock.")

if __name__ == '__main__':
    main()
