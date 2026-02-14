import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_and_save_model(csv_path='data/market_data.csv'):
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    # 2. Feature Engineering (The indicators the AI looks at)
    df['price_return'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    df['volatility'] = df['price_return'].rolling(window=10).std()
    df = df.dropna()

    # 3. Choose features for the model
    features = ['price_return', 'vol_change', 'volatility']
    X = df[features]

    # 4. Train Isolation Forest
    # contamination=0.015 means we assume 1.5% of data is "weird"
    model = IsolationForest(n_estimators=200, contamination=0.015, random_state=42)
    model.fit(X)

    # 5. Save the "Brain" and the Feature List
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/anomaly_detector.joblib')
    joblib.dump(features, 'models/feature_list.joblib')
    
    print(" Model trained on CSV and saved to /models")

if __name__ == "__main__":
    train_and_save_model()