"""
This strategy is based on the fact that, when there is a sudden change in trend it will  continue
more than one day.

So we need to look for a falling trend or a buying trend continuous for atleast 3-4 days
and then there is a change in trend. wait for one day to confirm
now check these in a day candle. and select the stocks

On the day of trading , open a 5 min candle .
In the first 5 min candle we need to check if its green , we buy 5-10 paisa over the High of first candle
and the mid of the candle is the stoploss and the range of the first candle is the target.
"""


import pandas as pd
import glob
import os


# dict to store all the dataframe
stock_df = {}
stock_list = []


# init stuff
BALANCE = 2000
init_bal = BALANCE
target_percent = 10
stoploss_percent = 7.5
stop_trading = False


# -------------------- Create and load Instances ------------------

historical_dir = "G:\AI Trading\Code\RayTrader_v3\Backtest\\ray_trader_mk1\hist_1_data\\*.csv"
file_end = "-EQ.csv"


# Func to load the csv and return a dataframe
def csv_to_df(csv_file):
    df = pd.read_csv(csv_file,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'])

    # Drop the header
    df = df.drop(df.index[0])
    # parse date
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %I:%M:%S %p')
    # Set index
    df = df.set_index('Date')
    # reindex the df to make it look right
    df = df.reindex(index=df.index[::-1])
    index = pd.to_datetime("10-04-2019 03:29:00 PM", format='%d-%m-%Y %I:%M:%S %p')
    df = df[:index]
    # print(df.head())
    # print(df.tail())
    return df


def select_stocks(start_index,end_index):
    global stock_df, stock_list
    selected_stocks = []

    # check which stocks follow these the day before
    # Open = Low or Open = High
    day_index = ""
    for df_stock in stock_list:
        df = stock_df[df_stock]
        df = df[start_index:end_index]
        df_high = df['High'].max()
        df_low = df['Low'].min()
        df_open = df['Open'].iloc[0]

        if df_open == df_high:
            tup = (df_stock, 'S')
            selected_stocks.append(tup)
        elif df_open == df_low:
            tup = (df_stock, 'B')
            selected_stocks.append(tup)
        day_index = str(df.index[0])
        # print("Stock: ",df_stock)
        # print(df.head())
        # print(df.tail())
        # print("High:",df_high,"Low",df_low,"Open",df_open)
    print(day_index)
    return selected_stocks


def runner_code():
    each_day = int(375)
    total_len = int(15375)
    total_days = int(total_len/each_day)
    index = 0
    for day in range(total_days):
        # Select stock for the day
        stocks = select_stocks(start_index=index, end_index=index+each_day)
        print(stocks)
        index += each_day
        # break

# load hist data into separate dict
data_dir = glob.glob(historical_dir)

for f in data_dir:
    stock = os.path.split(f)[-1].split("-")[0]
    stock_list.append(stock)
    stock_df[stock] = csv_to_df(f)

runner_code()
