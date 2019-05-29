"""
This code basically generates all the inputs from the historical data for neat-trader:

CandleStick Patterns:
1.Three Line Strike || 2.Three Black Crows || 3.Evening Star || 4.Abandoned Baby ||
5.Harami pattern || 6.Harami cross pattern || 7.Engulfing Pattern || 8.Hammer ||
9.Inverted Hammer || 10.Piercing Pattern

Technical Indicators:
1.EMA(5),(10),(20) || 2.SMA(5),(10),(20) || 3.Hull MA(9) || 4.Volume Weighted MA(20)
5.BETA() || 6.RSI(14),(8) || 7.ADX(14) || 8.Momentum(10) || 9.Macd(12,26,9)
10.Awesome Osc || 11.Stochastic(14,3,3) || 12.Stochastic Fast
13.Stochastic Relative Strength Index(3,3,14,14) || 14.Ultimate Oscillator(7,14,28)
15.Williams' %R(14) || 16.TSF || 17.STDDEV || 18.VAR || 19.CCI
"""

import pandas as pd
import talib
import math
from numpy import array
from collections import OrderedDict


# Func to load the csv and return a data-frame
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
    # convert all data
    df['Open'] = pd.to_numeric(df['Open'])
    df['High'] = pd.to_numeric(df['High'])
    df['Low'] = pd.to_numeric(df['Low'])
    df['Close'] = pd.to_numeric(df['Close'])
    df['Vol'] = pd.to_numeric(df['Vol'])
    # print(df.head())
    # print(df.tail())
    return df


def round_up(num_to_round, multiple):
    if multiple == 0:
        return num_to_round

    remainder = abs(num_to_round) % multiple
    if remainder == 0:
        return num_to_round

    if num_to_round < 0:
        return -(abs(num_to_round) - remainder)
    else:
        return num_to_round + multiple - remainder


def split_data(data, train_per=80):
    train_size = round_up(int(round(train_per / 100 * data.shape[0])),375)
    train = data[:train_size]
    test = data[train_size:]
    return train,test


def get_inputs(signals, timestep):
    input_signals = []
    for s in signals.values():
        input_signals.append(round(s[timestep],8))

    return input_signals


def awesome_osc(high, low, s=5, len=34):
    """Awesome Oscillator
    MEDIAN PRICE = (HIGH+LOW)/2
    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)
    """
    mp = 0.5 * (high + low)
    ao = talib.SMA(mp,s) - talib.SMA(mp,len)
    return ao


def get_hma(price, timeperiod=14):
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


def get_signals(data, tech_in=True, patterns=True):
    # dict to store signals
    signals = OrderedDict()

    # Technical Indicators
    if tech_in:
        signals["ema_5_by_10"] = talib.EMA(data['Close'], 5) / talib.EMA(data['Close'], 10)
        signals["ema_10_by_20"] = talib.EMA(data['Close'], 10) / talib.EMA(data['Close'], 20)
        signals["sma_5_by_10"] = talib.SMA(data['Close'], 5) / talib.SMA(data['Close'], 10)
        signals["sma_10_by_20"] = talib.SMA(data['Close'], 10) / talib.SMA(data['Close'], 20)
        signals["hma_9_by_18"] = get_hma(data['Close'], 9) / get_hma(data['Close'], 18)
        signals["bop"] = talib.BOP(data['Open'], data['High'], data['Low'], data['Close'])
        signals["beta"] = talib.BETA(data['High'], data['Low'])
        signals["rsi"] = talib.RSI(data['Close'])
        signals["adi"] = talib.ADX(data['High'], data['Low'], data['Close'])
        signals["natr"] = talib.NATR(data['High'], data['Low'], data['Close'])
        signals["mom"] = talib.MOM(data['Close'])
        macd, macdsignal, macdhist = talib.MACD(data['Close'])
        signals["macd"] = macd
        signals["macdsignal"] = macdsignal
        signals["macdhist"] = macdhist
        fastk, fastd = talib.STOCHRSI(data['Close'])
        signals["fastk"] = fastk
        signals["fastd"] = fastd
        signals["ulti"] = talib.ULTOSC(data['High'], data['Low'], data['Close'])
        signals["awesome"] = awesome_osc(data['High'],data['Low'])
        signals["wills_r"] = talib.WILLR(data['High'], data['Low'], data['Close'])

    if patterns:
        # pattern recognition
        signals["three_line_strike"] = talib.CDL3LINESTRIKE(data['Open'], data['High'], data['Low'], data['Close'])

        signals["three_black_crows"] = talib.CDL3BLACKCROWS(data['Open'], data['High'], data['Low'], data['Close'])

        signals["doji_star"] = talib.CDLDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])

        signals["evening_doji_star"] = talib.CDLEVENINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])

        signals["morning_doji_star"] = talib.CDLMORNINGDOJISTAR(data['Open'], data['High'], data['Low'], data['Close'])

        signals["morning_star"] = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])

        signals["evening_star"] = talib.CDLEVENINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])

        signals["shooting_star"] = talib.CDLSHOOTINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])

        signals["engulfing_patt"] = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])

        signals["hammer"] = talib.CDLHAMMER(data['Open'], data['High'], data['Low'], data['Close'])

        signals["inverted_hammer"] = talib.CDLINVERTEDHAMMER(data['Open'], data['High'], data['Low'], data['Close'])

        signals["hanging_man"] = talib.CDLHANGINGMAN(data['Open'], data['High'], data['Low'], data['Close'])

        signals["harami"] = talib.CDLHARAMI(data['Open'], data['High'], data['Low'], data['Close'])

        signals["harami_cross"] = talib.CDLHARAMICROSS(data['Open'], data['High'], data['Low'], data['Close'])

        signals["piercing"] = talib.CDLPIERCING(data['Open'], data['High'], data['Low'], data['Close'])

    return signals
