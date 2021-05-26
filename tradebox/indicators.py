from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
from config.secreats import ALPHA_API_KEY
import Data as DS
import pandas as pd


# Establishing a connection with AAlpha vantage technical indicator endpoint and output format
ti = TechIndicators(key=ALPHA_API_KEY, output_format='pandas')

def get_rsi(symbol, interval, period):
    data_rsi = ti.get_rsi(symbol=symbol, interval=interval, time_period=period, series_type='close')
    return data_rsi

def get_sma(symbol, interval, period):
    data_sma = ti.get_sma(symbol=symbol, interval=interval, time_period=period, series_type='close')
    return data_sma

def get_ema(symbol, interval, period):
    data_ema = ti.get_ema(symbol=symbol, interval=interval, time_period=period, series_type='close')
    return data_ema

def get_macd(symbol, interval, period):
    data_macd = ti.get_macd(symbol=symbol, interval=interval, time_period=period, series_type='close')
    return data_macd


def plot_rsi_vs_close(symbol, interval, period):
    # get rsi
    rsi_data = get_rsi(symbol, interval, period)
    # get timeseries
    ts_data = DS.get_timeseries(symbol, interval, 'interval')

    dataframe_1 = rsi_data
    dataframe_2 = ts_data['4. close'].iloc[period -1::]

    #group data for plot
    total_dataframe = pd.concat([dataframe_1, dataframe_2], axis=1)
    total_dataframe.plot()
    plt.show()

