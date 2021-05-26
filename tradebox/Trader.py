from tradebox import Data as dd
from tradebox import Machine_learning as ml
# from tradebox import indicators as indicator

dates = ml.dates
prices = ml.prices

# get active stocks
stocks = dd.get_active_stocks()
print(stocks)

# get stock data
data = dd.get_stock_data(sym= stocks,interval= '5min', typ= 'interval')
dd.save_to_excel(data, stocks)
print(data)

# alert user on if percentage goes above of 0.0004
dd.alert_user(0.0004, data)

# ML
# get SVM prediction
main_data = ml.get_data_excel()
predicted_prices =  ml.predict_prices_svm(dates, prices, 26)
print(predicted_prices)


# Plot RSI