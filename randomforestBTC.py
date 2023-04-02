import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tkinter import Tk, Label, Entry, Button, StringVar

# Enter your Binance API credentials
api_key = 'YOUR_API_KEY'
api_secret = 'YOUR_API_SECRET'
client = Client(api_key, api_secret)


def fetch_data():
    # Fetch BTC price data from the Binance API
    time_frame = '1h'
    now = datetime.now()
    end_time = now - timedelta(minutes=now.minute %
                               60, seconds=now.second, microseconds=now.microsecond)
    start_time = end_time - timedelta(days=365)
    klines = client.get_historical_klines(
        'BTCUSDT', time_frame, str(start_time), str(end_time))

    # Prepare the data
    btc_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    btc_df = btc_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    btc_df[['open', 'high', 'low', 'close', 'volume']] = btc_df[[
        'open', 'high', 'low', 'close', 'volume']].astype(float)
    btc_df.set_index('timestamp', inplace=True)

    return btc_df


def generate_features_and_targets(df):
    # Calculate the percentage change
    df['price_change'] = df['close'].pct_change()

    # Generate target labels: 1 = BUY, -1 = SELL, 0 = HOLD
    df['target'] = np.where(df['price_change'] > 0.005,
                            1, np.where(df['price_change'] < -0.005, -1, 0))

    # Calculate technical indicators (moving averages)
    df['ma_10'] = df['close'].rolling(window=10).mean()
    df['ma_30'] = df['close'].rolling(window=30).mean()

    # Drop missing values
    df.dropna(inplace=True)

    # Prepare features and targets
    features = df[['close', 'ma_10', 'ma_30']]
    targets = df['target']

    return features, targets


def train_random_forest(features, targets):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42)

    # Create a Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_clf.fit(X_train, y_train)

    # Test the classifier
    y_pred = rf_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Random Forest Classifier Accuracy: {accuracy}')

    return rf_clf


def create_gui(clf, df):
    # Create a simple GUI using tkinter
    import tkinter as tk

    def get_signal():
        latest_data = df.iloc[-1][['close',
                                   'ma_10', 'ma_30']].values.reshape(1, -1)
        signal = clf.predict(latest_data)[0]
        signal_str = 'HOLD' if signal == 0 else (
            'BUY' if signal == 1 else 'SELL')
        signal_label.config(text=f'Signal: {signal_str}')

    root = tk.Tk()
    root.title('BTC Trading Signal')

    signal_label = tk.Label(root, text='Signal:', font=('Arial', 24))
    signal_label.pack(pady=20)

    get_signal_button = tk.Button(
        root, text='Get Signal', command=get_signal, font=('Arial', 16))
    get_signal_button.pack(pady=10)

    root.mainloop()


if __name__ == '__main__':
    btc_df = fetch_data()
    features, targets = generate_features_and_targets(btc_df)
    classifier = train_random_forest(features, targets)
    create_gui(classifier, btc_df)
