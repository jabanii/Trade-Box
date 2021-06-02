from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from tradebox import Data as dd
import matplotlib.pyplot as plt
from config.secreats import ALPHA_API_KEY
from tradebox import Data as DS
import pandas as pd
import datetime
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

period = 60


# Establishing a connection with AAlpha vantage technical indicator endpoint and output format
ti = TechIndicators(key=ALPHA_API_KEY, output_format='pandas')
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

def get_rsi():
    symbol = input('RSI for What stock ?')
    interval = input("\n what interval would you like to get? '1min', '5min', '15min', '30min', '60min' \n")
    data_rsi = ti.get_rsi(symbol=symbol.upper(), interval=interval, time_period=period, series_type='close')
    print(data_rsi)
    return data_rsi

def get_sma():
    symbol = input('SMA for What stock ?')
    interval = input("\n what interval would you like to get? '1min', '5min', '15min', '30min', '60min' \n")
    data_sma = ti.get_sma(symbol=symbol.upper(), interval=interval, time_period=period, series_type='close')
    print(data_sma)
    return data_sma

def get_ema():
    symbol = input('EMA for What stock ? \n')
    interval = input("\n what interval would you like to get? '1min', '5min', '15min', '30min', '60min' \n")
    data_ema = ti.get_ema(symbol=symbol.upper(), interval=interval, time_period=period, series_type='close')
    print(data_ema)
    return data_ema

def get_macd():
    symbol = input('MACD for What stock ?')
    interval = input("\n what interval would you like to get? '1min', '5min', '15min', '30min', '60min' \n")
    data_macd = ti.get_macd(symbol=symbol.upper(), interval=interval, time_period=period, series_type='close')
    return data_macd


def plot_rsi_vs_close():
    symbol = input('RSI for What stock ?')
    interval = input("\n what interval would you like to get? '1min', '5min', '15min', '30min', '60min' \n")
    # get rsi
    ti_data = ti.get_rsi(symbol=symbol.upper(), interval=interval, time_period=period, series_type='close')
    # get timeseries
    ts_data = ts.get_intraday(symbol=symbol.upper(), interval=interval, outputsize="full")

    dataframe_1 = ti_data
    dataframe_2 = ts_data[2].iloc[period - 1::]
    #dataframe_2 = ts_data['4. close'].iloc[period -1::]

    dataframe_2.index = dataframe_1.index
    dataframe_2 = int(dataframe_2)

    #group data for plot
    total_dataframe = pd.concat([dataframe_1, dataframe_2], axis=1)
    print(total_dataframe)
    #total_dataframe.plot()
    #plt.show()


def rsi_dataframe():
    stock = input('What stock do you want?')
    period = 60
    ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
    data_ts = ts.get_intraday(stock.upper(), interval='1min', outputsize='full')

    ti = TechIndicators(key=stock.upper(), output_format='pandas')

    data_ti, meta_data_ti = ti.get_bbands(symbol=stock.upper(), interval='1min', time_period=period, series_type='close')


    df = data_ts[0][period::]

    # df.index = pd.Index(map(lambda x: str(x)[:-3], df.index))

    df2 = data_ti


    total_df = pd.merge(df,  df2, on="date")

    low = []
    for l in total_df['3. low']:
        low.append(float(l))

    high = []
    for h in total_df['2. high']:
        high.append(float(h))

    bb_low = []
    for bl in total_df['Real Lower Band']:
        bb_low.append(float(bl))

    bb_high = []
    for bh in total_df['Real Upper Band']:
        bb_high.append(float(bh))


    buy = []
    buy_index = []

    for bl, p, i in zip(bb_low, low, total_df.index[::-1]):
        if p < bl:
            if not buy_index:
                buy.append(p)
                buy_index.append(i)
            else:
                index_need_to_beat = buy_index[-1] + datetime.timedelta(minutes=30)
                if i > index_need_to_beat:
                    buy.append(p)
                    buy_index.append(i)

    # If Price signals a good sell

    sell = []
    sell_index = []
    for bh, p, i in zip(bb_high, high, total_df.index[::-1]):
        if p < bh:
            if not sell_index:
                sell.append(p)
                sell_index.append(i)
            else:
                index_need_to_beat = sell_index[-1] + datetime.timedelta(minutes=30)
                if i > index_need_to_beat:
                    sell.append(p)
                    sell_index.append(i)

    buy_positions = 0
    profit = 0
    stocks = 0
    buy_point = 0
    sell_point = 0

    while buy_point != len(buy):
        if buy_index[buy_point] < sell_index[sell_point]:
            buy_positions += round(float(buy[buy_point]))
            print(f'buy position = {buy[buy_point]} total positions = {round(buy_positions, 2)} at sell index = {sell_index[sell_point]}')
            buy_point += 1
            stocks += 1
        else:
            print(f'sold at {sell[sell_point]}')
            profit += buy_positions - (float(sell[sell_point]) * stocks)
            profit = round(profit, 2)
            print(f'profit = {profit}')
            print('')
            buy_positions = 0
            stocks =0
            sell_point += 1
    else:
        pass

    # for h in total_df.head():
    #     print(h)
    return print(f'${profit}')

def rsi_dash():

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    api_key = ALPHA_API_KEY
    app = dash.Dash()

    app.layout = html.Div([
        html.Link(
            rel='stylesheet',
            href='https://codepen.io/chriddyp/pen/bWLwgP.css'
        ),
        dcc.Input(id='input-box', value='', type='text', placeholder='Enter a Stock symbol', ),
        html.Button('Submit', id='button'),
        html.Div(),
        html.P('5 Calls Per Min'),
        dcc.Graph(
            id='candle-graph', animate=True, style={"backgroundColor": "#1a2d46", 'color': '#ffffff'}, ),
        html.Div([
            html.P('Developed by: ', style={'display': 'inline', 'color': 'white'}),
            html.A('Paul Lassa'),
            html.P(' - ', style={'display': 'inline', 'color': 'white'}),
            html.A('paullassa01@gmail.com', href='mailto:paullassa01@gmail.com')
        ], className="twelve columns",
            style={'fontSize': 18, 'padding-top': 20}

        )
    ])

    api_key = ALPHA_API_KEY
    period = 60
    ts = TimeSeries(key=api_key, output_format='pandas')
    ti = TechIndicators(key=api_key, output_format='pandas')

    @app.callback(Output('candle-graph', 'figure'),
                  [Input('button', 'n_clicks')],
                  [State('input-box', 'value')])
    def update_layout(n_clicks, input_value):

        # Getting Dataframes from Alphavantage
        data_ts = ts.get_intraday(symbol=input_value.upper(), interval='1min', outputsize='full')
        data_ti, meta_data_ti = ti.get_rsi(symbol=input_value.upper(), interval='1min', time_period=period,
                                           series_type='close')

        df = data_ts[0][period::]

        df.index = pd.Index(map(lambda x: str(x)[:-3], df.index))

        df2 = data_ti

        total_df = pd.concat([df, df2], axis=1, sort=True)

        # Breaking Down Datafames

        opens = []
        for o in total_df['1. open']:
            opens.append(float(o))

        high = []
        for h in total_df['2. high']:
            high.append(float(h))

        low = []
        for l in total_df['3. low']:
            low.append(float(l))

        close = []
        for c in total_df['4. close']:
            close.append(float(c))

        rsi_offset = []

        for r, l in zip(total_df['RSI'], low):
            rsi_offset.append(l - (l / r))

        # SELL SCATTER
        high_rsi_value = []
        high_rsi_time = []

        for value, time, l in zip(total_df['RSI'], total_df.index, low):
            if value > 60:
                high_rsi_value.append(l - (l / value))
                high_rsi_time.append(time)

        # BUY SCATTER
        low_rsi_value = []
        low_rsi_time = []

        for value, time, l in zip(total_df['RSI'], total_df.index, low):
            if value < 35:
                low_rsi_value.append(l - (l / value))
                low_rsi_time.append(time)

        scatter = go.Scatter(
            x=high_rsi_time,
            y=high_rsi_value,
            mode='markers',
            name='Sell'
        )
        scatter_buy = go.Scatter(
            x=low_rsi_time,
            y=low_rsi_value,
            mode='markers',
            name='Buy'
        )

        rsi = go.Scatter(
            x=total_df.index,
            y=rsi_offset,
        )

        BuySide = go.Candlestick(
            x=total_df.index,
            open=opens,
            high=high,
            low=low,
            close=close,
            increasing={'line': {'color': '#00CC94'}},
            decreasing={'line': {'color': '#F50030'}},
            name='candlestick'
        )
        data = [BuySide, rsi, scatter, scatter_buy]

        layout = go.Layout(
            paper_bgcolor='#27293d',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(type="category"),
            yaxis=dict(range=[min(rsi_offset), max(high)]),
            font=dict(color='white'),

        )
        return {'data': data, 'layout': layout}

        app.run_server(port=8085)

