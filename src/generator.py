import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_market_data(days=30):
    np.random.seed(42)
    periods = days * 24  # Hourly data
    base_time = datetime.now() - timedelta(days=days)
    
    data = {
        'timestamp': [base_time + timedelta(hours=i) for i in range(periods)],
        'price': np.linspace(100, 110, periods) + np.random.normal(0, 0.5, periods),
        'volume': np.random.normal(1000, 200, periods),
        'trades_count': np.random.randint(50, 150, periods)
    }
    
    df = pd.DataFrame(data)

    # Inject Anomaly 1: A "Pump" (Price + Volume Spike)
    df.loc[100:105, 'price'] *= 1.5
    df.loc[100:105, 'volume'] *= 5
    
    # Inject Anomaly 2: A "Flash Crash"
    df.loc[400:402, 'price'] *= 0.7
    df.loc[400:402, 'volume'] *= 3

    df.to_csv('data/market_data.csv', index=False)
    print("✅ Synthetic data generated in data/market_data.csv")

if __name__ == "__main__":
    generate_market_data()