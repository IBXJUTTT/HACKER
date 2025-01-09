import ccxt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# Function to fetch Forex data from an exchange
def fetch_forex_data(symbol, timeframe='1m', limit=1440):
    exchange = ccxt.kraken()  # Adjust as needed for different exchanges
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    
    # Add UTC+5 time column
    df['time_utc_plus_5'] = df.index + pd.Timedelta(hours=5)
    
    return df

# Function to create features for the AI model
def create_features(df):
    df['return'] = df['close'].pct_change()  # Daily return as feature
    df['rsi'] = compute_rsi(df['close'], window=14)  # RSI indicator
    df['ema'] = df['close'].ewm(span=20).mean()  # Exponential moving average (EMA)
    df['obv'] = compute_obv(df['close'], df['volume'])  # On-Balance Volume (OBV)
    df.dropna(inplace=True)  # Drop rows with NaN values
    return df

# Compute the Relative Strength Index (RSI)
def compute_rsi(series, window):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Compute On-Balance Volume (OBV) as a measure of buyer/seller pressure
def compute_obv(close, volume):
    obv = [0]
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv.append(obv[-1] + volume[i])  # Buyer pressure
        elif close[i] < close[i-1]:
            obv.append(obv[-1] - volume[i])  # Seller pressure
        else:
            obv.append(obv[-1])  # No change
    return pd.Series(obv, index=close.index)

# Function to generate buy/sell signals with TP and SL
def generate_signals_with_tp_sl(df, tp_ratio=0.01, sl_ratio=0.005):
    df['short_ma'] = df['close'].rolling(window=10).mean()
    df['long_ma'] = df['close'].rolling(window=50).mean()
    df['signal'] = 0
    df['signal'][10:] = np.where(df['short_ma'][10:] > df['long_ma'][10:], 1, -1)
    df['position'] = df['signal'].diff()
    df['take_profit'] = np.where(df['signal'] == 1, df['close'] * (1 + tp_ratio), df['close'] * (1 - tp_ratio))
    df['stop_loss'] = np.where(df['signal'] == 1, df['close'] * (1 - sl_ratio), df['close'] * (1 + sl_ratio))
    return df

# Function to include fundamental analysis placeholder
def add_fundamental_analysis(df):
    df['fundamental_score'] = np.random.randint(0, 100, size=len(df))  # Dummy score for fundamental analysis
    return df

# Prepares data for training the model
def prepare_data(df):
    X = df[['return', 'rsi', 'ema', 'obv', 'volume', 'fundamental_score']]  # Features
    y = np.where(df['return'].shift(-1) > 0, 1, 0)  # Target: 1 for 'up', 0 for 'down'
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()  # Scale features
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

# Train the Random Forest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict the signal (buy/sell) for the test data
def predict_signal(model, X_test):
    return model.predict(X_test)  # 1 for buy, 0 for sell

# Fetch all available currency pairs from Kraken
def fetch_currency_pairs():
    exchange = ccxt.kraken()  # Adjust as needed for different exchanges
    markets = exchange.load_markets()  # Load markets (currency pairs)
    pairs = [market for market in markets if '/USD' in market]  # Select pairs involving USD
    return pairs

# Streamlit app structure
def main():
    st.title("Forex Trading Signals - Currency Pairs")
    
    # Fetch the list of all available currency pairs
    pairs = fetch_currency_pairs()
    
    # User input: Select a currency pair from the list
    symbol = st.selectbox('Select Currency Pair:', pairs)
    
    # User input: Select timeframe for data
    timeframe = st.selectbox('Select Timeframe:', ['1m', '5m', '15m', '30m', '1h'])
    
    # Fetch data and process it
    st.write(f"Fetching data for {symbol} at {timeframe} timeframe...")
    df = fetch_forex_data(symbol, timeframe)
    df = create_features(df)
    df = add_fundamental_analysis(df)
    df = generate_signals_with_tp_sl(df)
    
    # Show the data
    st.subheader('Latest Data and Signals')
    st.write(df.tail())  # Display the latest rows
    
    # Prepare data for model training
    X_train, X_test, y_train, y_test = prepare_data(df)
    model = train_model(X_train, y_train)
    
    # Predict signals
    y_pred = predict_signal(model, X_test)
    
    # Add the predicted signal to the DataFrame
    df_predicted = df.iloc[-len(y_pred):].copy()
    df_predicted['predicted_signal'] = y_pred
    df_predicted['predicted_signal'] = df_predicted['predicted_signal'].map({1: 'Buy', 0: 'Sell'})
    df_predicted['take_profit'] = np.where(df_predicted['predicted_signal'] == 'Buy', df_predicted['close'] * 1.01, df_predicted['close'] * 0.99)
    df_predicted['stop_loss'] = np.where(df_predicted['predicted_signal'] == 'Buy', df_predicted['close'] * 0.995, df_predicted['close'] * 1.005)
    
    # Display the predictions
    st.subheader('Predicted Signals')
    st.write(df_predicted[['close', 'predicted_signal', 'take_profit', 'stop_loss']].tail())
    
    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader('Model Performance')
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text(classification_report(y_test, y_pred))

    # Display real-time signal every minute
    st.subheader('Real-Time Prediction')
    if st.button('Predict Now'):
        latest_signal = df_predicted['predicted_signal'].iloc[-1]
        latest_tp = df_predicted['take_profit'].iloc[-1]
        latest_sl = df_predicted['stop_loss'].iloc[-1]
        st.write(f"Latest signal for {symbol}: {latest_signal}")
        st.write(f"Take Profit: {latest_tp:.2f}, Stop Loss: {latest_sl:.2f}")

if __name__ == "__main__":
    main()
