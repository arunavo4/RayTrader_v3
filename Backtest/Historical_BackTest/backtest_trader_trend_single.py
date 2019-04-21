"""
@########################### Same Trend Strategy But With Single Stock ###############################

This strategy is based on the fact that, when there is a sudden change in trend it will  continue
more than one day.

So we need to look for a falling trend or a buying trend continuous for at least 3-4 days
and then there is a change in trend. wait for one day to confirm
now check these in a day candle. and select the stocks

On the day of trading , open a 5 min candle .
In the first 5 min candle we need to check if its green , we buy 5-10 paisa over the High of first candle
and the mid of the candle is the stop-loss and the range of the first candle is the target.

"""


import pandas as pd
import glob
import os


# dict to store all the dataframe
stock_df_day = {}
stock_df_5min = {}
stock_df_min = {}
stock_list = []


# init stuff
BALANCE = 2000
init_bal = BALANCE
target_percent = 10
stoploss_percent = 7.5
stop_trading = False


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


def data_resample(df_orgi, sample_size):
    # file_new = './Min_10_data/' + os.path.split(file_name)[-1]
    # sample_size = '10Min'
    data = df_orgi.copy()
    # data = data.convert_objects(convert_numeric=True)
    data['Open'] = pd.to_numeric(data['Open'])
    data['High'] = pd.to_numeric(data['High'])
    data['Low'] = pd.to_numeric(data['Low'])
    data['Close'] = pd.to_numeric(data['Close'])
    data['Vol'] = pd.to_numeric(data['Vol'])
    # print(data.dtypes)
    # print("1Min data",data.head())

    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Vol': 'sum'}

    # adding the base 15 adds an offset to the time
    data = data.resample(sample_size, base=15).apply(ohlc_dict).dropna(how='any')

    cols = ['Open', 'High', 'Low', 'Close', 'Vol']
    data = data[cols]
    # reindex it to look it like others
    # data = data.reindex(index=data.index[::-1])
    # print("15 Min data",data.head())
    # data.index = data.index.strftime('%d-%m-%Y %I:%M:%S %p')
    # data.index.name = 'Date'
    # data.to_csv(file_new, header=True)
    
    return data


def check_trend(df):
    direction = []

    for i in range(df.shape[0]):
        data = df.iloc[i]
        if data['Open'] > data['Close']:
            direction.append('-')
        else:
            direction.append('+')

    # Now u need to find 3 consecutive +++ or ---
    dir_str = ''.join(direction[:4])

    if dir_str.find('---+')==0:
        return True
    elif dir_str.find('+++-')==0:
        return True

    return False


def runner_code():
    # Start from 3 day later, I need to check and shortlist those
    # stocks that was going in a dir in past 3 days and now its moving
    # in opp dir so that's our stock

    init_index_day = 5
    each_day = int(375)
    total_len = int(15375)
    total_days = int(total_len/each_day)
    for day in range(total_days-init_index_day):
        end_index = init_index_day
        start_index = init_index_day - 5
        init_index_day += 1
        date = ''
        selected_stocks = []
        for stock in stock_list:
            df = stock_df_day[stock].copy()
            date = df.index[end_index]
            data = df[start_index:end_index]
            if check_trend(data):
                # Now we can trade in this stock
                selected_stocks.append(stock) 

        print("Day:",date)
        print(selected_stocks)
        start_date_time = pd.to_datetime(date, format='%d-%m-%Y %H:%M:%S')
        start_date_time = start_date_time.to_pydatetime().replace(hour=9,minute=15)
        end_date_time = pd.to_datetime(date, format='%d-%m-%Y %H:%M:%S')
        end_date_time = end_date_time.to_pydatetime().replace(hour=15, minute=25)

        # Now check the 5 min candle data to decide Buy or Sell
        for stock in selected_stocks:
            df_5_Min = stock_df_5min[stock].copy()
            data_5 = df_5_Min[start_date_time:end_date_time]

        break

#------------------------ Main Code ---------------------------------------------------

historical_dir = "G:\AI Trading\Code\RayTrader_v3\Backtest\\ray_trader_mk1\hist_1_data\\*.csv"

# load hist data into separate dict
# Basically we load data into 1min and then resample for the day and 5 min
data_dir = glob.glob(historical_dir)

for f in data_dir:
    stock = os.path.split(f)[-1].split("-")[0]
    stock_list.append(stock)
    stock_df_min[stock] = csv_to_df(f)
    stock_df_5min[stock] = data_resample(stock_df_min[stock],'5Min')
    stock_df_day[stock] = data_resample(stock_df_min[stock],'D')


runner_code()