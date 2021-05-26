import csv
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xlrd
import Data as ds
# dependent and independent data arrays
dates = []
prices = []
xl_file = 'AMCoutput.xlsx'

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            dates.append(int(row[0]))

            print(dates)
            prices.append(float(row[4]))
        return

def get_data_excel():
    df = pd.read_excel(xl_file, sheet_name=None)
    for row in df:
        dates.append(int(row[0]))
        prices.append(float(row[4]))
    return

# .split('-')[0]

def predict_prices_svm(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel='linear', C=1e3).fit(dates, prices)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    plt.scatter(dates, prices, color='black', label='data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='green', label='linear model')
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='polynomial model')
    plt.xlabel('Dates')
    plt.ylabel('Price')
    plt.title('Support Vector Regresion')
    plt.show()
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]


def predict_price_LSTM(data):
    # get data
    symbols = 'MD'
    #stock_data = ds.get_stock_data(sym=symbols, interval='1min', typ='daily')
    stock_data=data

    # reverse rows
    stock_data = stock_data.loc[::-1, :]

    print(stock_data.head())

    # check for missing  missing data and cleaninfg
    # returns bollean if not applicable returns false if okay
    checker = stock_data.isna().any()
    print(checker)

    # get info of data
    # all data is float so no need to homoginise it
    info = stock_data.info()
    print(info)

    # get 7 day rolling mean(moving average)
    # rolling_mean = stock_data.rolling(7).mean().head(20)
    # print(rolling_mean.head(20))
    # stock_data['4. close'].plot(figsize = (16,6))
    # stock_data.rolling(window=30).mean()['4. close'].plot()
    # plt.show()
    training_set = stock_data.loc[:, ['4. close', '2. high']]

    # data pre-processin

    #  feature scaling normalization
    sc = MinMaxScaler(feature_range=(0, 1))

    training_set_scaled = sc.fit_transform(training_set)

    print(training_set_scaled)
    # creating a dataset with 60 timesteps and 1 output
    x_train = []
    y_train = []
    stock_index = stock_data.index

    for i in range(60, len(stock_index)):
        x_train.append(training_set_scaled[i - 60:i, 0])
        y_train.append(training_set_scaled[1, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # reshape data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # build the lstm
    # feature extraction
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # initialise RNN
    regressor = Sequential()

    # training the neural network
    # data is fed to the NN
    # here we assign biases and weights
    # This model composes of a sequential input layer,
    # 3 lstm layers and
    # a dense layer with activation,
    # dense output layer with linear activation function

    # add first Lstm layer
    # dropout is a regularisation technique for reducing overfitting in Neural Networks
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Adding fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))

    # adding output layer
    # onlu one output is needed so units = 1
    regressor.add(Dense(units=1))

    # compiling the RNN
    # here we will use the Adam optimiser
    # this will affect how fast the algorithm converges to the minimum value,
    regressor.compile(optimizer='adam', loss='mean_squared_error')

    # fitting the RNN to the training set
    regressor.fit(x_train, y_train, epochs=100, batch_size=32)

    # visualisation
    # Predicting Future Stock using the Test Set
    stock_data_test = stock_data
    real_stock_price = stock_data.loc[:, ['4. close', '2. high']]

    # to make prdiction
    # Merge the training set and the test set on the 0 axis.
    # Set the time step as 60 (as seen previously)
    # Use MinMaxScaler to transform the new dataset
    # Reshape the dataset as done previously
    # After making the predictions we use inverse_transform to get back the stock prices in normal readable format.
    dataset_total = pd.concat((stock_data['4. close'], stock_data['2. high']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(stock_data_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, 76):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # plot results
    plt.plot(real_stock_price, color='black', label='Stock Price')
    plt.plot(predicted_stock_price, color='green', label='Predicted Stock Price')
    plt.title('Stock Price Prediction LSTM')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()



get_data_excel()
predicted_price = predict_prices_svm(dates, prices, 26)
print(predicted_price)


