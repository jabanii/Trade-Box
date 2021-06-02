from tradebox import Data as dd
from tradebox import Machine_learning as ml
from tradebox import indicators as ind
import sys
# from tradebox import indicators as indicator

dates = ml.dates
prices = ml.prices
#stocks = ""
period = 60

# find relevant stocks
# Stream realtime quotes (alert on sudden changes)
        # show data and Graphs
# Technical analysis (alert on sudden changes)
            # show graphs for technical indicators and plot against each other
# get stock predictions (using lstm or using SVM)

def menue():
    ans = True
    while ans:
        print("*********** Welcome to Tradebox **************")
        print("""
        1. Find Relevant Stocks
        2. Stream Realtime Quotes
        3. Technical Analysis
        4. Get Stock Predictions 
        5. Trade using indicator
        6. Exit/Quit 
        """)

        choice = input("what would you like to do today?\n")

        if choice == "1":
            print('\n Todays most active stocks stocks are: ')
            stocks = dd.get_active_stocks()
            print(stocks)

        elif choice == "2":
            stream = input("\n Input the symbol of the stock you would like to stream \n")
            interval = input("\n what interval would you like to get? '1min', '5min', '15min', '30min', '60min' \n")
            data = dd.get_stock_data(sym= stream,interval= interval, typ= 'interval')
            #todo: saving info to excel
            #dd.save_to_excel(data, stocks)
            print(data)
            # alert user on if percentage goes above of 0.0004
            dd.alert_user(0.0004, data)

        elif choice == "3":
            print("""
            1. RSI (Relative Strength Index)
            2. SMA (Simple Moving Average)
            3. EMA (Exponential Moving Average)
            4. MACD (Moving Average Convergence Difference)
            """)
            indicator_choice = input("what technical indicators would you like to see? \n")
            if indicator_choice == "1":
                print('\n Get RSI (Relative Strength Index) for: ')
                ind.get_rsi()
                #ind.plot_rsi_vs_close()

            elif indicator_choice == "2":
                print('\n Get SMA (Simple Moving Average) for: ')
                ind.get_sma()

            elif indicator_choice == "3":
                print('\n Get EMA (Exponential Moving Average) for: ')
                ind.get_ema()

            elif indicator_choice == "4":
                print('\n Get MACD (Moving Average Convergence Difference for: ')
                ind.get_macd()

        elif choice == "4":
            print("\n What Machine learning model do you want to use for Prediction")
            print("""
                    1. get stock predictions using SVM (support vector machine)
                    2. get stock predictions using LSTM (Long short-term memory)
                    """)
            ml_choice = input("\n Model: \n")
            if ml_choice == "1":
                print('\n Get SVM (support vector machine) prediction for: ')
                ml.predict_prices_svm()

            elif ml_choice == "2":
                print('\n Get LSTM (Long short-term memory) prediction for: ')
                ml.predict_price_LSTM()
            else:
                print('\n Not Valid Choice Try again')

        elif choice == "5":
            print("\n Trade with RSI")
            ind.rsi_dataframe()

        elif choice == "6":
            print("\n Goodbye")
            sys.exit

        elif choice != "":
            print("\n Not Valid Choice Try again")

menue()