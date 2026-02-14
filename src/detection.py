from sklearn.ensemble import IsolationForest
import joblib

def train_anomaly_detector(df):
    # Features to look at
    features = ['price_return', 'vol_change', 'volatility']
    X = df[features]
    
    # contamination=0.02 means we expect roughly 2% anomalies
    model = IsolationForest(contamination=0.02, random_state=42)
    df['anomaly_signal'] = model.fit_predict(X)
    
    # Calculate "Risk Score" (Distance from normality)
    # Mapping decision_function to 0-100 scale
    scores = model.decision_function(X)
    df['risk_score'] = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100
    
    joblib.dump(model, 'models/anomaly_model.joblib')
    return df