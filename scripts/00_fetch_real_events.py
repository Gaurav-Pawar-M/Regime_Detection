import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os

# Unique list of 19 stocks (TCS.NS was repeated in the prompt, counting uniquely we have 18, 
# but if including it, we'll just process the unique list)
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

START_DATE = pd.to_datetime("2019-01-01")
END_DATE = pd.to_datetime("2026-07-04")

def fetch_screener(symbol):
    """Scrapes screener.in for quarterly results, board meetings, and AGMs."""
    short_sym = symbol.replace('.NS', '')
    url = f"https://www.screener.in/company/{short_sym}/consolidated/"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    events = []
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Check quarters table and announcements list
        # We parse the generic announcements feed because Screener dynamically lists exact dates there.
        # Alternatively, the quarters table headers (like 'Mar 2023') are extracted as a fallback Results proxy
        
        announcements_ul = soup.find('ul', id='announcements')
        if announcements_ul:
            for li in announcements_ul.find_all('li'):
                text = li.text.lower()
                event_type = None
                
                if 'result' in text:
                    event_type = "Results"
                elif 'board meeting' in text:
                    event_type = "Board Meeting"
                elif 'agm' in text or 'annual general meeting' in text:
                    event_type = "AGM"
                
                if event_type:
                    # The date is usually printed before the link or inside a span
                    date_div = li.find('div', class_='date') or li.find('span')
                    if date_div:
                        try:
                            date_str = date_div.text.strip()
                            dt = pd.to_datetime(date_str)
                            events.append({
                                'symbol': symbol,
                                'event_date': dt,
                                'event_type': event_type
                            })
                        except Exception:
                            pass
                            
        # Look for the quarters table dates (fallback for Results if announcements failed)
        if not events:
            quarters_table = soup.find('section', id='quarters')
            if quarters_table:
                ths = quarters_table.find_all('th')
                for th in ths:
                    th_text = th.text.strip()
                    if th_text:
                        try:
                            # parse 'Mar 2023' as end of month proxy for the result quarter
                            dt = pd.to_datetime(th_text) + pd.offsets.MonthEnd(0)
                            events.append({
                                'symbol': symbol,
                                'event_date': dt,
                                'event_type': "Results"
                            })
                        except Exception:
                            pass

        return pd.DataFrame(events) if events else pd.DataFrame(columns=['symbol', 'event_date', 'event_type'])
    except Exception as e:
        return pd.DataFrame(columns=['symbol', 'event_date', 'event_type'])

def fetch_bse(symbol):
    """Fallback query to BSE API if Screener fails."""
    bse_code = BSE_MAPPING.get(symbol)
    if not bse_code: 
        return pd.DataFrame(columns=['symbol', 'event_date', 'event_type'])
    
    url = "https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    params = {"scripcode": bse_code}
    
    events = []
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        
        # Process the JSON API response (if it successfully returns an API payload)
        if response.headers.get('content-type', '').startswith('application/json'):
            data = response.json()
            if isinstance(data, list):
                for item in data:
                    subject = str(item.get('Subject', '')).lower() + str(item.get('Headline', '')).lower()
                    date_str = item.get('NEWS_DT') or item.get('DT_TM')
                    
                    event_type = None
                    if 'result' in subject:
                        event_type = "Results"
                    elif 'board meeting' in subject:
                        event_type = "Board Meeting"
                    elif 'agm' in subject or 'annual general' in subject:
                        event_type = "AGM"
                    
                    if event_type and date_str:
                        dt = pd.to_datetime(date_str)
                        events.append({
                            'symbol': symbol,
                            'event_date': dt,
                            'event_type': event_type
                        })
    except Exception:
        pass
    
    return pd.DataFrame(events) if events else pd.DataFrame(columns=['symbol', 'event_date', 'event_type'])

def main():
    all_events = []
    
    print("Starting data extraction...")
    for i, symbol in enumerate(STOCKS, 1):
        try:
            df = fetch_screener(symbol)
            source = "Screener"
            
            if df.empty:
                df = fetch_bse(symbol)
                source = "BSE fallback"
            
            if not df.empty:
                df['event_date'] = pd.to_datetime(df['event_date'])
                df = df[(df['event_date'] >= START_DATE) & (df['event_date'] <= END_DATE)]
                
            n_events = len(df)
            if n_events > 0:
                print(f"[{i}/{len(STOCKS)}] {symbol}: Found {n_events} events ({source})")
                all_events.append(df)
            else:
                print(f"[{i}/{len(STOCKS)}] {symbol}: WARNING - Both Screener and BSE failed to find events.")
                
            # Polite scraping delay
            time.sleep(2)
        except Exception as e:
            # Catch all exception so it never crashes the script per stock
            print(f"[{i}/{len(STOCKS)}] {symbol}: ERROR - {str(e)}")

    if all_events:
        final_df = pd.concat(all_events, ignore_index=True)
        
        # Sort and drop duplicates as requested
        final_df = final_df.sort_values(by='event_date', ascending=True)
        final_df = final_df.drop_duplicates(subset=['symbol', 'event_date'])
        
        # Cast to datetime64[ms]
        final_df['event_date'] = final_df['event_date'].astype('datetime64[ms]')
        
        # Safe pathing to ensure data is dumped cleanly to the correct place
        out_path = os.path.join('data', 'events_table_real.parquet')
        
        # Ensure data folder exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        final_df.to_parquet(out_path, index=False)
        
        total = len(final_df)
        mean_events = total / len(STOCKS)
        
        print("\nSUMMARY")
        print(f"Total events: {total}")
        print(f"Events per stock (mean): {mean_events:.1f}")
        print(f"Date range: {final_df['event_date'].min().strftime('%Y-%m-%d')} to {final_df['event_date'].max().strftime('%Y-%m-%d')}")
        print(f"Saved to ../{out_path}")
        
        if total < 50:
            print("WARNING: Very few events scraped")
    else:
        print("\nWARNING: No events scraped at all!")

if __name__ == '__main__':
    main()
