import pandas as pd

def clean_and_feature_engineer(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate % changes (Recruiters love domain-specific features)
    df['price_return'] = df['price'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    df['volatility'] = df['price_return'].rolling(window=5).std()
    
    return df.dropna()