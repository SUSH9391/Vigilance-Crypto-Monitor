import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta, UTC

def fetch_historical_ohlcv(symbol='BTC/USDT', timeframe='1h', days_back=60):
    exchange = ccxt.binance({'enableRateLimit': True})
    
    since = exchange.parse8601((datetime.now(UTC) - timedelta(days=days_back)).isoformat())
    all_ohlcv = []
    
    print(f" Fetching {days_back} days of data for {symbol}...")

    while since < exchange.milliseconds():
        try:
            # Fetch batch
            new_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not new_ohlcv:
                break
            
            # Update 'since' for next iteration
            since = new_ohlcv[-1][0] + 1
            all_ohlcv.extend(new_ohlcv)
            
            print(f" Progress: {len(all_ohlcv)} candles collected...")
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f" API Error: {e}")
            break

    # 1. CREATE DATAFRAME
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 2. DROP DUPLICATES (Critical for pagination safety)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

    # 3. GAP CHECKING (Fintech best practice)
    # Ensure there are no missing hours. Missing data = Bad ML training.
    expected_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='h')
    if len(df) < len(expected_range):
        print(f" Warning: Detected {len(expected_range) - len(df)} missing candles. Filling with last known values.")
        df = df.set_index('timestamp').reindex(expected_range).ffill().reset_index()
        df.rename(columns={'index': 'timestamp'}, inplace=True)

    # 4. EXPORT
    output_path = 'data/market_data.csv'
    df.to_csv(output_path, index=False)
    print(f" Cleaned data saved ({len(df)} rows).")
    
    return df

if __name__ == "__main__":
    fetch_historical_ohlcv()