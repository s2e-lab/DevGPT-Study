import tkinter as tk
from tkinter import ttk
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def fetch_data():
    ticker = ticker_entry.get()
    if not ticker:
        result_label.config(text="Please enter a stock ticker.")
        return

    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period="1y")
        if hist_data.empty:
            result_label.config(text="Invalid stock ticker or no data available.")
            return

        display_data(hist_data)
        generate_summary(stock)
        result_label.config(text=f"Showing data for {ticker}")

    except Exception as e:
        result_label.config(text=f"An error occurred: {str(e)}")

def display_data(data: pd.DataFrame):
    fig, axs = plt.subplots(4, 1, figsize=(14, 14))

    # Historical Closing Prices
    axs[0].plot(data['Close'])
    axs[0].set_title('Historical Stock Price')
    axs[0].set_xlabel('Date')
    axs[0].set_ylabel('Close Price (USD)')

    # Moving Averages
    short_window = 40
    long_window = 100
    signals = pd.DataFrame(index=data.index)
    signals['price'] = data['Close']
    signals['short_ma'] = signals['price'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['long_ma'] = signals['price'].rolling(window=long_window, min_periods=1, center=False).mean()
    axs[1].plot(signals[['price', 'short_ma', 'long_ma']])
    axs[1].set_title('Moving Averages')
    axs[1].set_xlabel('Date')
    axs[1].set_ylabel('Price (USD)')
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    axs[2].plot(rsi)
    axs[2].axhline(0, linestyle='--', alpha=0.5)
    axs[2].axhline(70, linestyle='--', alpha=0.5)
    axs[2].axhline(30, linestyle='--', alpha=0.5)
    axs[2].set_title('Relative Strength Index (RSI)')
    axs[2].set_xlabel('Date')
    axs[2].set_ylabel('RSI')

    # Volume
    axs[3].plot(data['Volume'])
    axs[3].set_title('Volume')
    axs[3].set_xlabel('Date')
    axs[3].set_ylabel('Volume')

    plt.tight_layout()
    plt.show()

# More code remains the same...
