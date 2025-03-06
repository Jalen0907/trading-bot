import ccxt
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ------------------------------
# Step 1: Fetch Historical Data
# ------------------------------

def fetch_ohlcv(symbol="BTC/USDT", timeframe="1h", limit=100):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ------------------------------
# Step 2: Preprocess Data
# ------------------------------

def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['close']])
    
    X, y = [], []
    for i in range(10, len(df_scaled)):
        X.append(df_scaled[i-10:i, 0])  # Last 10 price points
        y.append(df_scaled[i, 0])  # Next price prediction
    
    return np.array(X), np.array(y), scaler

# ------------------------------
# Step 3: Build AI Model
# ------------------------------

def create_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(10, 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ------------------------------
# Step 4: Train Model
# ------------------------------

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)
    return model

# ------------------------------
# Step 5: Make Predictions & Trade
# ------------------------------

def make_trade(symbol="BTC/USDT", risk_factor=0.01):
    exchange = ccxt.binance({
        'apiKey': 'YOUR_API_KEY',
        'secret': 'YOUR_SECRET_KEY',
        'enableRateLimit': True
    })

    df = fetch_ohlcv(symbol)
    X, _, scaler = preprocess_data(df)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    prediction = model.predict(X[-1].reshape(1, 10, 1))
    predicted_price = scaler.inverse_transform(prediction)[0, 0]

    last_price = df['close'].iloc[-1]
    
    if predicted_price > last_price:
        print(f"Buying {symbol} at {last_price}")
        exchange.create_market_buy_order(symbol, risk_factor * exchange.fetch_balance()['USDT']['total'])
    else:
        print(f"Selling {symbol} at {last_price}")
        exchange.create_market_sell_order(symbol, risk_factor * exchange.fetch_balance()[symbol.split('/')[0]]['total'])

# ------------------------------
# Main Execution
# ------------------------------

if __name__ == "__main__":
    df = fetch_ohlcv()
    X, y, scaler = preprocess_data(df)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_model()
    model = train_model(model, X, y)

    while True:
        make_trade()
        time.sleep(3600)  # Run every hour
