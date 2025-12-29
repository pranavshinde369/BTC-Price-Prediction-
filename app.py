import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from flask import Flask, render_template
from tensorflow.keras.models import load_model
import datetime
import os

app = Flask(__name__)

#Global Variables 
model = None
scaler = None

def load_assets():
    global model, scaler
    try:
        # Check if files exist first
        if not os.path.exists('btc_model.keras'):
            print("ERROR: model file not found.")
            return
        
        model = load_model('btc_model.keras')
        scaler = joblib.load('btc_scaler.joblib')
        print("Model and Scaler loaded successfully!")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")

load_assets()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return render_template('index.html', prediction_text="Error: Model not loaded. Run train.py first!")

    try:
        # 1. FETCH LIVE DATA
        stock = "BTC-USD"
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=100)
        
        print(f"Fetching data for {stock}...")
        df = yf.download(stock, start=start_date, end=end_date, progress=False)
        
        # --- FIX STARTS HERE ---
        # Fix for yfinance returning MultiIndex columns (e.g. Price, Ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Ensure we have the 'Close' column
        if 'Close' not in df.columns:
            print("Columns found:", df.columns)
            return render_template('index.html', prediction_text="Error: 'Close' column not found in data.")
        # --- FIX ENDS HERE ---

        if len(df) < 60:
            return render_template('index.html', prediction_text="Error: Not enough data fetched.")

        # 2. PREPARE DATA
        # Use simple bracket access which is safer now
        data = df[['Close']] 
        last_60_days = data[-60:].values
        
        # Check shape before scaling
        if last_60_days.shape[1] == 0:
             return render_template('index.html', prediction_text="Error: Data processing failed (Empty columns).")

        # Scale
        last_60_days_scaled = scaler.transform(last_60_days)
        
        # Reshape [Samples, Time Steps, Features]
        X_test = []
        X_test.append(last_60_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # 3. PREDICT
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)
        result = float(pred_price[0][0])
        
        current_price = float(data.iloc[-1]['Close'])
        
        # Calculate date for tomorrow
        tomorrow = end_date + datetime.timedelta(days=1)
        
        return render_template('index.html', 
                               prediction_text=f"${result:,.2f}",
                               last_price=f"${current_price:,.2f}",
                               date_tomorrow=tomorrow.strftime('%Y-%m-%d'))

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)