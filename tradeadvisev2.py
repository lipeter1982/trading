import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from talib import abstract
from datetime import datetime, timedelta

# Enter your Binance API credentials
api_key = 'uMSreSVTUwzR8t9hIU63DVUVGnMdJicE8YSSz9E6qfAxJ23LYYefDP1OfBuGEUq0'
api_secret = 'C1pnvdgdjm7RlkCG3kIikU7fi5i6xmA59iT9kzOrO7NWo2uQn3E0yrX9WldYAX6k'
client = Client(api_key, api_secret)

# Define the time frames for the BTC chart
time_frames = ['15m', '30m', '1h']

# Define the technical indicators to use
indicators = ['ma', 'bbands', 'rsi', 'macd', 'adx']

# Define the scalping parameters
buy_threshold = 30
sell_threshold = 70
stop_loss = 0.45
bbands_period = 20
bbands_dev = 1.5

# Fetch BTC price data from the Binance API
now = datetime.now()
end_time = now - timedelta(minutes=now.minute %
                           5, seconds=now.second, microseconds=now.microsecond)
start_time = end_time - timedelta(hours=24)
btc_data = {}
for tf in time_frames:
    klines = client.get_historical_klines(
        'BTCUSDT', tf, str(start_time), str(end_time))
    btc_df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                           'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    btc_df = btc_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
    btc_df[['open', 'high', 'low', 'close', 'volume']] = btc_df[[
        'open', 'high', 'low', 'close', 'volume']].astype(float)
    btc_df.set_index('timestamp', inplace=True)
    btc_data[tf] = btc_df

# Calculate the technical indicators
for tf in time_frames:
    for indicator in indicators:
        if indicator == 'ma':
            btc_data[tf]['ma'] = abstract.MA(
                btc_data[tf]['close'], timeperiod=20, matype=0)
        elif indicator == 'bbands':
            upperband, middleband, lowerband = abstract.BBANDS(
                btc_data[tf]['close'], timeperiod=20, nbdevup=float(bbands_dev), nbdevdn=float(bbands_dev), matype=0)
            btc_data[tf]['upperband'] = upperband
            btc_data[tf]['middleband'] = middleband
            btc_data[tf]['lowerband'] = lowerband
        elif indicator == 'rsi':
            btc_data[tf]['rsi'] = abstract.RSI(
                btc_data[tf]['close'], timeperiod=14)
        elif indicator == 'macd':
            macd, signal, hist = abstract.MACD(
                btc_data[tf]['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            btc_data[tf]['macd'] = macd
            btc_data[tf]['signal'] = signal
            btc_data[tf]['hist'] = hist
        elif indicator == 'adx':
            btc_data[tf]['adx'] = abstract.ADX(btc_data[tf], timeperiod=14)

# Create the BTC chart for all timeframes
fig, axs = plt.subplots(len(time_frames), 1, figsize=(10, 5*len(time_frames)))

for idx, tf in enumerate(time_frames):
    ax = axs[idx]
    ax.set_title(f'BTC Price Chart ({tf})')
    ax.plot(btc_data[tf]['close'], label='Price')

    # Plot the Bollinger Bands and MACD
    ax.plot(btc_data[tf]['upperband'], label='Upper Band')
    ax.plot(btc_data[tf]['middleband'], label='Middle Band')
    ax.plot(btc_data[tf]['lowerband'], label='Lower Band')
    ax.legend()


plt.show()
