import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import ccxt
import pandas as pd
import joblib
from preprocessing import clean_and_feature_engineer

# Initialize App with a Dark Theme (Slate/Cyborg)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
exchange = ccxt.binance()

# Load the "Saved Brain"
model = joblib.load('models/anomaly_detector.joblib')
features = joblib.load('models/feature_list.joblib')

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Vigilance Crypto Monitor", className="text-center text-primary mb-4"), width=12)
    ]),
    
    # KPI Row: Real-time Stats
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Current Price", className="card-title"),
                html.H2(id="live-price", className="text-success")
            ])
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Risk Score", className="card-title"),
                html.H2(id="risk-indicator")
            ])
        ]), width=4),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.H4("Status", className="card-title"),
                html.H2(id="market-status")
            ])
        ]), width=4),
    ], className="mb-4"),

    # Controls & Graph
    dbc.Row([
        dbc.Col([
            html.Label("Select Asset:"),
            dcc.Dropdown(
                id='asset-dropdown',
                options=[
                    {'label': 'Bitcoin (BTC)', 'value': 'BTC/USDT'},
                    {'label': 'Ethereum (ETH)', 'value': 'ETH/USDT'},
                    {'label': 'Solana (SOL)', 'value': 'SOL/USDT'}
                ],
                value='BTC/USDT',
                className="mb-3 text-dark"
            ),
            dcc.Interval(id='update-interval', interval=15000, n_intervals=0), # 15s updates
            dcc.Graph(id='live-graph')
        ], width=12)
    ])
], fluid=True)

@app.callback(
    [Output('live-graph', 'figure'), 
     Output('live-price', 'children'),
     Output('risk-indicator', 'children'),
     Output('market-status', 'children'),
     Output('risk-indicator', 'className')],
    [Input('update-interval', 'n_intervals'),
     Input('asset-dropdown', 'value')]
)
def update_dashboard(n, selected_asset):
    # 1. LIVE PIPELINE: Pull latest data
    ohlcv = exchange.fetch_ohlcv(selected_asset, timeframe='1h', limit=100)
    raw_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 2. PREPROCESS
    df = clean_and_feature_engineer(raw_df)

    # 3. PREDICT
    scores = model.decision_function(df[features])
    # Normalize score (Inverted decision_function where more negative = anomaly)
    df['risk_score'] = (1 - (scores - scores.min()) / (scores.max() - scores.min())) * 100
    
    latest_price = df['close'].iloc[-1]
    latest_risk = df['risk_score'].iloc[-1]
    
    # 4. VISUALIZE
    fig = px.line(df, x='timestamp', y='close', template="plotly_dark", title=f"Real-Time {selected_asset}")
    
    # Add Red Dots for Anomalies (> 80 Risk)
    anomalies = df[df['risk_score'] > 80]
    fig.add_scatter(x=anomalies['timestamp'], y=anomalies['close'], 
                    mode='markers', marker=dict(color='red', size=12), name='ANOMALY FLAG')

    # Styling Logic
    risk_color = "text-danger" if latest_risk > 80 else "text-warning" if latest_risk > 50 else "text-success"
    status_text = "🚨 ALERT: ANOMALY" if latest_risk > 80 else "✅ STABLE"

    return fig, f"${latest_price:,.2f}", f"{latest_risk:.1f}%", status_text, risk_color

if __name__ == '__main__':
    app.run(debug=True)