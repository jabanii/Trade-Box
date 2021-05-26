from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from td.client import TDClient
from config.secreats import ALPHA_API_KEY
from config.secreats import CONSUMER_KEY, REDIRECT_URI, JSON_PATH, TD_ACCOUNT, EMAIL_PASSWORD, SENDER_EMAIL
import requests
import smtplib, ssl
from datetime import datetime
import math
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import time

# format for output from Alpha vantage api
outputFormat = 'pandas'
outputSize = 'full'

# reciever email
rec_email = 'pajaniworks@gmail.com'

#time
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
market_close = '16:30'

# Connect to time series and Technical Indicator endpoint on Alpha vantage
ts = TimeSeries(key=ALPHA_API_KEY, output_format=outputFormat)
ti = TechIndicators(key=ALPHA_API_KEY, output_format=outputFormat)


# Connect to TD ameritrade.
# create instance of client
# td_client = TDClient(client_id=CONSUMER_KEY, redirect_uri=REDIRECT_URI,account_number=None,credentials_path=JSON_PATH)

# login to new session
# td_client.login()


def get_active_stocks():
    # create empty list for company tickers/symbol
    company_tickers = []
    # load webpage content
    URL = 'https://finance.yahoo.com/gainers'
    page = requests.get(URL)

    # convert to beautiful soup object
    soup = BeautifulSoup(page.content, features="html.parser")
    symbol = soup.find('a', attrs={'class': 'Fw(600) C($linkColor)'})

    # append tickers to list
    company_tickers = symbol.string
    # return list of tickers
    return company_tickers


# This method calls on the Alpha Vantage api to  get time series data for stock
# it takes the ticker symbol "sym"
# the interval, "interval" : '1min', '5min', '15min', '30min', '60min'
# the type of data, "typ" : 'daily', 'weekly', 'monthly' and 'interval'
# outputs a pandas data array of specified data
def get_stock_data(sym, interval, typ):
    if typ == 'daily':
        state = ts.get_daily_adjusted(sym,outputsize=outputSize)[0]
    elif typ == 'weekly':
        state = ts.get_weekly_adjusted(sym)[0]
    elif typ == 'monthly':
        state =ts.get_monthly_adjusted(sym)[0]
    elif typ == 'interval':
        state = ts.get_intraday(symbol=sym, interval=interval, outputsize=outputSize)[0]
    else:
        print('Wrong Entry Format')
    #print(state)
    return state

# saves gotten stock data to a csv
def save_to_excel(data, symb):
    i = 1
    while i == 1:
        data.to_excel(symb + "output.xlsx")
        #time.sleep(60) #appends data every minute
        #print(data)

# send mail
def send_mail(receiver_email, message):
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    sender_email = SENDER_EMAIL
    password = EMAIL_PASSWORD

    # Create a secure SSL context
    context = ssl.create_default_context()
    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()


# Alert User
def alert_user(percentage_change, data):
    close_data = data['4. close']
    pchange = close_data.pct_change()
    last_change = pchange[-1]

    if abs(last_change) > percentage_change:
        # print('Stock Alert' + last_change)
        message = 'Stock alert, %Change'

        send_mail(rec_email, message)

#symbols = 'MD'

#stock_data = get_stock_data(sym=symbols,interval='1min',typ='interval')

#save_to_excel(stock_data,symbols)
