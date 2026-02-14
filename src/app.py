from dash import Dash, dcc, html
import plotly.express as px
from preprocessing import clean_and_feature_engineer
from detection import train_anomaly_detector

app = Dash(__name__)

# Load and process data
# Ensure this matches the filename you downloaded from Kaggle
df = clean_and_feature_engineer('data/market_data.csv') 
df = train_anomaly_detector(df)

# Create Plotly Chart
# If your Kaggle data uses 'close', use y='close'. If 'price', use y='price'.
y_axis_col = 'close' if 'close' in df.columns else 'price'

fig = px.scatter(df, x='timestamp', y=y_axis_col, color='risk_score',
                 size='volume', color_continuous_scale='Reds',
                 title='Crypto Vigilance: Real-time Anomaly Detection')

# Add a horizontal line for high-risk threshold
fig.add_hline(y=df[y_axis_col].mean(), line_dash="dot", annotation_text="Average Price")

app.layout = html.Div([
    html.H1("Vigilance Crypto Monitor"),
    dcc.Graph(figure=fig),
    html.Div([
        html.H3("System Status: Monitoring Live Feed"),
        html.P(f"Latest Risk Score: {df['risk_score'].iloc[-1]:.2f}"),
        html.P("Note: Higher risk scores (Dark Red) indicate potential pump-and-dump anomalies.")
    ])
])

if __name__ == '__main__':
    # Updated from run_server to run
    app.run(debug=True)