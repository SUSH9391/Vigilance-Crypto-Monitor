import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import ccxt
import pandas as pd
import joblib

# Initialize App & CCXT
app = dash.Dash(__name__)
server = app.server # Essential for deployment
exchange = ccxt.binance()

# Load the "Saved Brain" from Stage 1
model = joblib.load('models/anomaly_detector.joblib')
features = joblib.load('models/feature_list.joblib')

app.layout = html.Div([
    html.H1("Vigilance Crypto: Live Anomaly Monitor"),
    dcc.Interval(id='update-interval', interval=30000, n_intervals=0), # Refresh every 30s
    dcc.Graph(id='live-graph'),
    html.Div(id='risk-status')
])

@app.callback(
    [Output('live-graph', 'figure'), Output('risk-status', 'children')],
    [Input('update-interval', 'n_intervals')]
)
def update_dashboard(n):
    # 1. LIVE PIPELINE: Pull latest 100 hours from Binance
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=100)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 2. Pre-process Live Data
    df['price_return'] = df['close'].pct_change()
    df['vol_change'] = df['volume'].pct_change()
    df['volatility'] = df['price_return'].rolling(window=10).std()
    df = df.dropna()

    # 3. PREDICT: Ask the AI for a Risk Score
    scores = model.decision_function(df[features])
    # Convert score to 0-100 (Higher is riskier)
    df['risk_score'] = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100

    # 4. Visualize
    fig = px.line(df, x='timestamp', y='close', title="Real-Time BTC/USDT Price")
    
    # Highlight anomalies in Red
    anomalies = df[df['risk_score'] > 80]
    fig.add_scatter(x=anomalies['timestamp'], y=anomalies['close'], 
                    mode='markers', marker=dict(color='red', size=10), name='ANOMALY')

    latest_score = df['risk_score'].iloc[-1]
    status = f"Current Market Risk: {latest_score:.2f}%"
    
    return fig, html.H3(status, style={'color': 'red' if latest_score > 80 else 'green'})

if __name__ == '__main__':
    app.run(debug=True)