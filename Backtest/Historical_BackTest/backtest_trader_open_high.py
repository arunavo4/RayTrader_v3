"""
@########################### Open=High or Open=Low Strategy ###############################
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


# Func to trade on 1 min data being pessimistic though
def trader(df, mode, price, stoploss, target):
    # In the most simple manner I am just going to check if
    # I hit target or stoploss and return profit/loss
    # print("Trading on 1 min Data")
    k = 0
    for i in range(df.shape[0]):
        data = df.iloc[i]
        # if i < 5:
        #     print(data)
        if mode == 'Buy':
            if float(data['High']) >= float(target):
                # print("Buy target reached")
                return 1
            elif float(data['Low']) <= float(stoploss):
                # print("Buy stoploss reached")
                return -1
            elif price < float(data['Close']):
                k=1
            elif price > float(data['Close']):
                k=-1

        elif mode == 'Sell':
            if float(data['High']) >= float(stoploss):
                # print("Sell stoploss reached")
                return -1
            elif float(data['Low']) <= float(target):
                # print("Sell target reached")
                return 1
            elif price > float(data['Close']):
                k=1
            elif price > float(data['Close']):
                k=-1

    if k==1:
        return 1
    elif k==-1:
        return -1
    else:
        return 0

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


def runner_code():
    # Start from 3 day later, I need to check and shortlist those
    # stocks that was going in a dir in past 3 days and now its moving
    # in opp dir so that's our stock
    init_index_day = 1
    total_days = stock_df_day[stock_list[0]].shape[0]
    total_win_loss = []
    for day in range(total_days - init_index_day):
        win_loss = []
        end_index = init_index_day
        start_index = init_index_day - 1
        init_index_day += 1
        date = ''
        selected_stocks = []
        selected_stocks_dict = {}
        for stock in stock_list:
            df = stock_df_day[stock].copy()
            try:
                date = df.index[end_index]
                data = df[start_index:end_index]
                prev_day = data.iloc[0]
                if prev_day['Open'] == prev_day['High']:
                    # print(stock)
                    # print(prev_day)
                    selected_stocks_dict[stock] = 'S'
                    selected_stocks.append(stock)
                elif prev_day['Open'] == prev_day['Low']:
                    # print(stock)
                    # print(prev_day)
                    selected_stocks_dict[stock] = 'B'
                    selected_stocks.append(stock)

            except:
                print(stock)
                print(total_days)
                print(df.shape[0])
                print(end_index)
        print("Day:", date)
        # print(selected_stocks)
        print(selected_stocks_dict)
        start_date_time = pd.to_datetime(date, format='%d-%m-%Y %H:%M:%S')
        start_date_time = start_date_time.to_pydatetime().replace(hour=9, minute=15)
        end_date_time = pd.to_datetime(date, format='%d-%m-%Y %H:%M:%S')
        end_date_time = end_date_time.to_pydatetime().replace(hour=15, minute=29)
        # Now check the 5 min candle data to decide Buy or Sell
        for stock in selected_stocks:
            data_1 = stock_df_min[stock].copy()
            data_1 = data_1[start_date_time:end_date_time]
            # check the first candle in 5min

            first_candle = data_1.iloc[0]

            out = 0
            if selected_stocks_dict[stock]=='B':
                # Red candle
                # print("\n\n Stock: ", stock)
                # print("First 1 min Candle:")
                # print(first_candle)
                stoploss = round(float(first_candle['Open'])+((0.3/100)*float(first_candle['Open'])), 2)
                target = round(float(first_candle['Open'])-((0.6/100)*float(first_candle['Open'])), 2)
                out = trader(data_1, 'Sell', float(first_candle['Open']), stoploss, target)
                win_loss.append(out)
                # print("Sell")
                # print("Price:", first_candle['Low'], "target:", target, "Stoploss:", stoploss)
            elif selected_stocks_dict[stock]=='S':
                # Green candle
                # print("\n\n Stock: ", stock)
                # print("First 1 min Candle:")
                # print(first_candle)
                stoploss = round(float(first_candle['Open']) - ((0.3 / 100) * float(first_candle['Open'])), 2)
                target = round(float(first_candle['Open']) + ((0.6 / 100) * float(first_candle['Open'])), 2)
                out = trader(data_1, 'Buy', float(first_candle['Open']), stoploss, target)
                win_loss.append(out)
                # print("Buy")
                # print("Price:", first_candle['High'], "target:", target, "Stoploss:", stoploss)

            # break
        print("Wins: ",win_loss.count(1),"\t","Loss: ",win_loss.count(-1))
        total_win_loss.extend(win_loss)
        # break
    print("\nTotal Wins and losses:")
    print("Wins: ", total_win_loss.count(1), "\t", "Loss: ", total_win_loss.count(-1))
    try:
        print("Win Loss Ratio:", round(int(total_win_loss.count(1))/int(total_win_loss.count(-1)),2))
    except ZeroDivisionError:
        if int(total_win_loss.count(1)) > 0:
            print("Win Loss Ratio:", round(int(total_win_loss.count(1)),2))
        else:
            print("Win Loss Ratio:", round(int(total_win_loss.count(-1)),2))


# ------------------------ Main Code ---------------------------------------------------

historical_dir = "G:\AI Trading\Code\RayTrader_v3\HistoricalData\Min_data_new\\*.csv"

# load hist data into separate dict
# Basically we load data into 1min and then resample for the day and 5 min
data_dir = glob.glob(historical_dir)

print("Loading Data")
for f in data_dir:
    try:
        stock = os.path.split(f)[-1].split("-")[0]
        stock_list.append(stock)
        stock_df_min[stock] = csv_to_df(f)
        stock_df_day[stock] = data_resample(stock_df_min[stock], 'D')
    except:
        print(f)

print("Data Loaded")

runner_code()