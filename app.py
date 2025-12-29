# =================================================
# APP.PY - The Flask Web Server
# =================================================
import numpy as np
import pandas as pd
import yfinance as yf # For LIVE data
import joblib
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime

app = Flask(__name__)

# GLOBAL VARIABLES
model = None
scaler = None

def load_assets():
    """Load model and scaler once when app starts"""
    global model, scaler
    try:
        model = load_model('btc_model.keras')
        scaler = joblib.load('btc_scaler.joblib')
        print("Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model/scaler. {e}")
        print("Did you run ipynb file first?")

load_assets()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text="Error: Model not loaded. Run train.py first!")

    try:
        # 1. FETCH LIVE DATA FROM YAHOO FINANCE
        # We need at least the last 60 days to make a prediction
        stock = "BTC-USD"
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=100) # Buffer to ensure we get 60 days
        
        print(f"Fetching live data for {stock}...")
        df = yf.download(stock, start=start_date, end=end_date, progress=False)
        
        if len(df) < 60:
            return render_template('index.html', prediction_text="Error: Could not fetch enough live data.")

        # 2. PREPARE DATA
        # Filter 'Close' price and get the last 60 days
        data = df.filter(['Close'])
        last_60_days = data[-60:].values
        
        # Scale the data (Must use the SAME scaler from training)
        last_60_days_scaled = scaler.transform(last_60_days)
        
        # Reshape for LSTM [Samples, Time Steps, Features]
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 3. PREDICT
        pred_price = model.predict(X_test)
        
        # Inverse transform to get actual USD price
        pred_price = scaler.inverse_transform(pred_price)
        result = float(pred_price[0][0])
        
        # Get the latest actual price for comparison
        current_price = float(data.iloc[-1]['Close'])
        
        return render_template('index.html', 
                               prediction_text=f"${result:,.2f}",
                               last_price=f"${current_price:,.2f}",
                               date_tomorrow=(end_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d'))

    except Exception as e:
        print(e)
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)