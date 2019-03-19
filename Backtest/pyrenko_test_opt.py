"""
    Pyrenko lib : https://github.com/quantroom-pro/pyrenko/blob/master/pyrenko_tutorial.ipynb
"""
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import talib
import scipy.optimize as opt
from statistics import mean

class renko:
    def __init__(self):
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.oc_2 = []

    # Setting brick size. Auto mode is preferred, it uses history
    def set_brick_size(self, HLC_history=None, auto=True, brick_size=10.0):
        if auto == True:
            self.brick_size = self.__get_optimal_brick_size(HLC_history.iloc[:, [0, 1, 2]])
        else:
            self.brick_size = brick_size
        return self.brick_size

    def __renko_rule(self, last_price):
        # Get the gap between two prices
        gap_div = int(float(float(last_price) - float(self.renko_prices[-1])) / self.brick_size)
        is_new_brick = False
        start_brick = 0
        num_new_bars = 0

        # When we have some gap in prices
        if gap_div != 0:
            # Forward any direction (up or down)
            if (gap_div > 0 and (int(self.renko_directions[-1]) > 0 or int(self.renko_directions[-1]) == 0)) or (
                    gap_div < 0 and (int(self.renko_directions[-1]) < 0 or int(self.renko_directions[-1]) == 0)):
                num_new_bars = gap_div
                is_new_brick = True
                start_brick = 0
            # Backward direction (up -> down or down -> up)
            elif np.abs(gap_div) >= 2:  # Should be double gap at least
                num_new_bars = gap_div
                num_new_bars -= np.sign(gap_div)
                start_brick = 2
                is_new_brick = True
                self.renko_prices.append(str(float(self.renko_prices[-1]) + 2 * float(self.brick_size) * int(np.sign(gap_div))))
                self.renko_directions.append(str(np.sign(gap_div)))
            # else:
            # num_new_bars = 0

            if is_new_brick:
                # Add each brick
                for d in range(start_brick, np.abs(gap_div)):
                    self.renko_prices.append(str(float(self.renko_prices[-1]) + float(self.brick_size) * int(np.sign(gap_div))))
                    self.renko_directions.append(str(np.sign(gap_div)))

        return num_new_bars

    # Getting renko on history
    def build_history(self, prices):
        if len(prices) > 0:
            # Init by start values
            self.source_prices = prices
            self.renko_prices.append(prices.iloc[0])
            self.renko_directions.append(0)

            # For each price in history
            for p in self.source_prices[1:]:
                self.__renko_rule(p)

        return len(self.renko_prices)

    # Getting next renko value for last price
    def do_next(self, last_price):
        if len(self.renko_prices) == 0:
            self.source_prices.append(last_price)
            self.renko_prices.append(last_price)
            self.renko_directions.append(0)
            return 1
        else:
            self.source_prices.append(last_price)
            return self.__renko_rule(last_price)

    # Simple method to get optimal brick size based on ATR
    def __get_optimal_brick_size(self, HLC_history, atr_timeperiod=14):
        brick_size = 0.0

        # If we have enough of data
        if HLC_history.shape[0] > atr_timeperiod:
            brick_size = np.median(talib.ATR(high=np.double(HLC_history.iloc[:, 0]),
                                             low=np.double(HLC_history.iloc[:, 1]),
                                             close=np.double(HLC_history.iloc[:, 2]),
                                             timeperiod=atr_timeperiod)[atr_timeperiod:])

        return brick_size

    def evaluate(self, method='simple'):
        balance = 0
        sign_changes = 0
        price_ratio = len(self.source_prices) / len(self.renko_prices)

        if method == 'simple':
            for i in range(2, len(self.renko_directions)):
                if self.renko_directions[i] == self.renko_directions[i - 1]:
                    balance = balance + 1
                else:
                    balance = balance - 2
                    sign_changes = sign_changes + 1

            if sign_changes == 0:
                sign_changes = 1

            score = balance / sign_changes
            if score >= 0 and price_ratio >= 1:
                score = np.log(score + 1) * np.log(price_ratio)
            else:
                score = -1.0

            return {'balance': balance, 'sign_changes:': sign_changes,
                    'price_ratio': price_ratio, 'score': score}

    def get_renko_prices(self):
        return self.renko_prices

    def get_renko_directions(self):
        return self.renko_directions

    def get_oc2_price(self):
        #get the mean of open close
        self.oc_2.clear()

        for i in range(1,len(self.renko_prices)):
            j=i-1
            self.oc_2.append((float(self.renko_prices[j])+float(self.renko_prices[i]))/2.0)

        return np.array(self.oc_2)

    def plot_renko(self, col_up='g', col_down='r'):
        fig, ax = plt.subplots(1, figsize=(20, 10))
        ax.set_title(name[-1])
        ax.set_xlabel('Renko bars')
        ax.set_ylabel('Price')

        self.renko_prices = [float(i) for i in self.renko_prices]
        self.renko_directions = [float(i) for i in self.renko_directions]

        # Calculate the limits of axes
        ax.set_xlim(0.0,
                    len(self.renko_prices) + 1.0)
        ax.set_ylim(np.min(self.renko_prices) - 3.0 * self.brick_size,
                    np.max(self.renko_prices) + 3.0 * self.brick_size)

        # Plot each renko bar
        for i in range(1, len(self.renko_prices)):
            # Set basic params for patch rectangle
            col = col_up if self.renko_directions[i] == 1 else col_down
            x = i
            y = self.renko_prices[i] - self.brick_size if self.renko_directions[i] == 1 else self.renko_prices[i]
            height = self.brick_size

            # Draw bar with params
            ax.add_patch(
                patches.Rectangle(
                    (x, y),  # (x,y)
                    1.0,  # width
                    self.brick_size,  # height
                    facecolor=col
                )
            )
        x_range = np.array([(i+0.5) for i in range(1,len(self.renko_prices))])
        plt.plot(x_range,sma)

        plt.show()


def ohlc_4(dataframe_l):
    ohlc_by_4  = []
    for i in range(len(dataframe_l)):
        ohlc_by_4.append((float(dataframe_l.Open[i]) + float(dataframe_l.High[i]) + float(dataframe_l.Low[i]) + float(dataframe_l.Close[i]))/4.0)
    return np.array(ohlc_by_4)

# Function for optimization
def evaluate_renko(brick, history, column_name):
    renko_obj = renko()
    renko_obj.set_brick_size(brick_size = brick, auto = False)
    renko_obj.build_history(prices = history)
    return renko_obj.evaluate()[column_name]

# Get a pandas dataframe
datapath = ('G:\AI Trading\Code\RayTrader_v3\Backtest\historical_data\POWERGRID-EQ.csv')
name = os.path.split(datapath)

df = pd.read_csv(datapath,
                       names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'])

#Drop the header
df = df.drop(df.index[0])
#parse date
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %I:%M:%S %p')
#Set index
df = df.set_index('Date')
#reindex the df to make it look right
df = df.reindex(index=df.index[::-1])

# Set a split date
date_split = '2018-11-01'
date_split_2 = '2018-11-19'
# df = df[date_split:date_split_2]

# Get ATR values (it needs to get boundaries)
# Drop NaNs
atr = talib.ATR(high = np.double(df.High),
                low = np.double(df.Low),
                close = np.double(df.Close),
                timeperiod = 14)

atr = atr[np.isnan(atr) == False]

# Get optimal brick size as maximum of score function by Brent's (or similar) method
# First and Last ATR values are used as the boundaries
optimal_brick_sfo = opt.fminbound(lambda x: -evaluate_renko(brick = x,
                                                            history = df.Close, column_name = 'score'),
                                  np.min(atr), np.max(atr), disp=0)

# Get optimal brick size based
# optimal_brick = renko().set_brick_size(auto = True, HLC_history = df[["High", "Low", "Close"]])
# optimal_brick = renko().set_brick_size(auto = False, brick_size = 1.4758)

# Build Renko chart
renko_obj_sfo = renko()
print('Set brick size to optimal: ', renko_obj_sfo.set_brick_size(auto = False, brick_size = optimal_brick_sfo))
renko_obj_sfo.build_history(prices = df.Close)
print('Renko bar prices: ', renko_obj_sfo.get_renko_prices())
print('Renko bar directions: ', renko_obj_sfo.get_renko_directions())
print('Renko bar evaluation: ', renko_obj_sfo.evaluate())

sma = talib.SMA(renko_obj_sfo.get_oc2_price(), timeperiod=8)
# sma = sma[np.isnan(sma) == False]

print('sma:', sma)

print('sma -1', sma[-1])

if len(renko_obj_sfo.get_renko_prices()) > 1:
    renko_obj_sfo.plot_renko()