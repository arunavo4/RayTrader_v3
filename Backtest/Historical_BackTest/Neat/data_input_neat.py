"""
This code basically has all the inputs needed for neat:

CandleStick Patterns:
1.Three Line Strike
2.Three Black Crows
3.Evening Star
4.Abandoned Baby
5.Harami pattern
6.Harami cross pattern
7.Engulfing Pattern
8.Hammer
9.Inverted Hammer
10.Piercing Pattern

Technical Indicators:
1.EMA(5),(10),(20)
2.SMA(5),(10),(20)
3.Hull MA(9)
4.Volume Weighted MA(20)
5.BETA()
6.RSI(14),(8)
7.ADX(14)
8.Momentum(10)
9.Macd(12,26,9)
10.Awesome Osc
11.Stochastic(14,3,3)
12.Stochastic Fast
13.Stochastic Relative Strength Index(3,3,14,14)
14.Ultimate Oscillator(7,14,28)
15.Williams' %R(14)
16.TSF
17.STDDEV
18.VAR
19.CCI
"""

import pandas as pd
import glob
import os
import talib
import math
from numpy import array


stock_df_min = {}
stock_list = []


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
    # convert all ddta
    df['Open'] = pd.to_numeric(df['Open'])
    df['High'] = pd.to_numeric(df['High'])
    df['Low'] = pd.to_numeric(df['Low'])
    df['Close'] = pd.to_numeric(df['Close'])
    df['Vol'] = pd.to_numeric(df['Vol'])
    # print(df.head())
    # print(df.tail())
    return df

def get_hma( price, timeperiod=14):
    # HMA= WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
    return (talib.WMA(
        2 * talib.WMA(price, timeperiod=math.floor(timeperiod / 2)) - talib.WMA(price, timeperiod=timeperiod),
        timeperiod=math.sqrt(timeperiod)))

def get_vwma(price, vol, timeperiod=14):
    # vwma = SUM( price * vol)/SUM(Vol)
    price_vol = array([a*b for a,b in zip(price,vol)])
    numerator = talib.SUM(price_vol, timeperiod)
    denominator = talib.SUM(vol, timeperiod)
    vwma = array([a/b for a,b in zip(numerator,denominator)])
    return vwma


def cal_inputs():

    for stock in stock_list:
        data = stock_df_min[stock]
        print("\n Stock: ", stock)

        # Technical Indicators
        ema_5 = talib.EMA(data['Close'],5)
        ema_10 = talib.EMA(data['Close'],10)
        ema_20 = talib.EMA(data['Close'],20)

        sma_5 = talib.SMA(data['Close'], 5)
        sma_10 = talib.SMA(data['Close'], 10)
        sma_20 = talib.SMA(data['Close'], 20)

        hma = get_hma(data['Close'],9)

        bop = talib.BOP(data['Open'],data['High'],data['Low'],data['Close'])

        vwma = get_vwma(data['Close'],data['Vol'],20)

        beta = talib.BETA(data['High'],data['Low'])
        rsi = talib.RSI(data['Close'])

        adi = talib.ADX(data['High'],data['Low'],data['Close'])
        natr = talib.NATR(data['High'],data['Low'],data['Close'])

        mom = talib.MOM(data['Close'])
        macd, macdsignal, macdhist = talib.MACD(data['Close'])

        fastk, fastd = talib.STOCHRSI(data['Close'])

        ulti = talib.ULTOSC(data['High'],data['Low'],data['Close'])

        chalkin_ad_osc = talib.ADOSC(data['High'],data['Low'],data['Close'],data['Vol'])
        wills_r = talib.WILLR(data['High'],data['Low'],data['Close'])

        print("ema_5:(last)", round(ema_5[-1],2))
        print("ema_10:(last)", round(ema_10[-1],2))
        print("ema_20:(last)", round(ema_20[-1],2))

        print("sma_5:(last)",round(sma_5[-1],2))
        print("sma_10:(last)",round(sma_10[-1],2))
        print("sma_20:(last)",round(sma_20[-1],2))

        print("HMA:(last)",round(hma[-1],2))
        print("BOP: (last)",round(bop[-1],2))
        print("Vol weighted MA:(last)",round(vwma[-1],2))
        print("BETA :(last)",round(beta[-1],2))
        print("RSI :(last)",round(rsi[-1],2))
        print("ADI:(last)",round(adi[-1],2))
        print("NATR:(last)",round(natr[-1],2))
        print("MOM:(last)",round(mom[-1],2))
        print("MACD :(last)",round(macd[-1],2),round(macdsignal[-1],2),round(macdhist[-1],2))
        print("STOCK RSI :(last)",round(fastk[-1],2),round(fastd[-1],2))
        print("Ultimate Osc: (last)",round(ulti[-1],2))
        print("chalkin_ad Osc: (last)",round(chalkin_ad_osc[-1],2))
        print("Williams: (last)",round(wills_r[-1],2))
        print("")

        # pattern recognition
        three_line_strike = talib.CDL3LINESTRIKE(data['Open'],data['High'],data['Low'],data['Close'])
        print("Three line Strike: ",next((x for x in three_line_strike if x != 0), None))
        print("Times Occured: ",len([x for x in three_line_strike if x != 0]))
        print("")

        three_black_crows = talib.CDL3BLACKCROWS (data['Open'], data['High'], data['Low'], data['Close'])
        print("three_black_crows: ", next((x for x in three_black_crows if x != 0), None))
        print("Times Occured: ",len([x for x in three_black_crows if x != 0]))
        print("")

        doji_star = talib.CDLDOJISTAR (data['Open'], data['High'], data['Low'], data['Close'])
        print("doji_star: ", next((x for x in doji_star if x != 0), None))
        print("Times Occured: ",len([x for x in doji_star if x != 0]))
        print("")

        evening_doji_star = talib.CDLEVENINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])
        print("evening_doji_star: ", next((x for x in evening_doji_star if x != 0), None))
        print("Times Occured: ", len([x for x in evening_doji_star if x != 0]))
        print("")

        morning_doji_star = talib.CDLMORNINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])
        print("morning_doji_star: ", next((x for x in morning_doji_star if x != 0), None))
        print("Times Occured: ", len([x for x in morning_doji_star if x != 0]))
        print("")

        morning_star = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        print("morning_star: ", next((x for x in morning_star if x != 0), None))
        print("Times Occured: ", len([x for x in morning_star if x != 0]))
        print("")

        evening_star = talib.CDLEVENINGSTAR (data['Open'], data['High'], data['Low'], data['Close'])
        print("evening_star: ", next((x for x in evening_star if x != 0), None))
        print("Times Occured: ",len([x for x in evening_star if x != 0]))
        print("")

        shooting_star = talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
        print("shooting_star: ", next((x for x in shooting_star if x != 0), None))
        print("Times Occured: ", len([x for x in shooting_star if x != 0]))
        print("")

        engulfing_patt = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
        print("engulfing_patt: ", next((x for x in engulfing_patt if x != 0), None))
        print("Times Occured: ",len([x for x in engulfing_patt if x != 0]))
        print("")

        hammer = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
        print("hammer: ", next((x for x in hammer if x != 0), None))
        print("Times Occured: ",len([x for x in hammer if x != 0]))
        print("")

        inverted_hammer = talib.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close'])
        print("inverted_hammer: ", next((x for x in inverted_hammer if x != 0), None))
        print("Times Occured: ",len([x for x in inverted_hammer if x != 0]))
        print("")

        hanging_man = talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])
        print("hanging_man: ", next((x for x in hanging_man if x != 0), None))
        print("Times Occured: ",len([x for x in hanging_man if x != 0]))
        print("")

        harami = talib.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])
        print("harami: ", next((x for x in harami if x != 0), None))
        print("Times Occured: ",len([x for x in harami if x != 0]))
        print("")

        harami_cross = talib.CDLHARAMICROSS(data['Open'], data['High'], data['Low'], data['Close'])
        print("harami_cross: ", next((x for x in harami_cross if x != 0), None))
        print("Times Occured: ",len([x for x in harami_cross if x != 0]))
        print("")

        piercing = talib.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])
        print("piercing: ", next((x for x in piercing if x != 0), None))
        print("Times Occured: ",len([x for x in piercing if x != 0]))
        print("")

# ------------------------ Main Code ---------------------------------------------------

historical_dir = "G:\AI Trading\Code\RayTrader_v3\HistoricalData\Min_data\\*.csv"

# load hist data into separate dict
# Basically we load data into 1min and then resample for the day and 5 min
data_dir = glob.glob(historical_dir)

print("Loading Data")
for f in data_dir:
    try:
        stock = os.path.split(f)[-1].split("-")[0]
        stock_list.append(stock)
        stock_df_min[stock] = csv_to_df(f)
    except:
        print(f)
    break
print("Data Loaded")

cal_inputs()
