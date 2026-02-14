import pandas as pd
import numpy as np

def clean_and_feature_engineer(df_or_path):
    """
    Standardizes and prepares crypto data for anomaly detection.
    Can accept a CSV path or a live Pandas DataFrame.
    """
    # 1. Load data if a path is provided
    if isinstance(df_or_path, str):
        df = pd.read_csv(df_or_path)
    else:
        df = df_or_path.copy()

    # 2. Standardize column names (fixes 'price' vs 'close' issues)
    df.columns = [col.lower() for col in df.columns]
    if 'close' in df.columns and 'price' not in df.columns:
        df['price'] = df['close']

    # 3. Time handling
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

    # 4. Feature Engineering (The "Hiring Manager" Features)
    # Log Returns are better than % change because they are additive and more stable
    df['price_return'] = np.log(df['price'] / df['price'].shift(1))
    
    # Volume Change
    df['vol_change'] = df['volume'].pct_change()
    
    # 10-period Volatility (measures "panic" in the market)
    df['volatility'] = df['price_return'].rolling(window=10).std()
    
    # Volume Intensity: Volume relative to its recent average
    df['vol_intensity'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # 5. Final Cleanup
    # Drop rows with NaN (from pct_change and rolling windows)
    return df.dropna().reset_index(drop=True)