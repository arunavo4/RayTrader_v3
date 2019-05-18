#
# Test strategies before launching
#
#   Renko + ATR + HMA + RSI Strategy with BO order
#

import pickle
import os, math
import glob
import numpy as np
import pandas as pd
import talib
import time
import threading
import multiprocessing

# Read the List of stocks to subscribe to
with open('G:\AI Trading\Code\RayTrader_v3\Backtest\symbol_token.out', 'rb') as fp:
    stock_token_dict = pickle.load(fp)

# Seperate Symbols and Tokens
stock_list = list(stock_token_dict.keys())

token_list = list(stock_token_dict.values())

token_list = [int(token) for token in token_list]

# Read the Optimal Renko Box size
with open('G:\AI Trading\Code\RayTrader_v3\Backtest\opt_renko_box.out', 'rb') as f:
    opt_renko_box = pickle.load(f)  # use `pickle.loads` to do the reverse

# Dict to store class objects to reference later
class_object_dict = {}


class RayTrader:

    def __init__(self, stock_name):
        self.stock_name = stock_name
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.oc_2 = []
        self.open = 0.0
        self.change = 0.0
        self.orders = []
        self.close = []
        self.long = False
        self.short = False
        self.pos = 0.0
        self.pos_qty = 1
        self.pnl = 0.0
        self.unrealized_pnl = 0.0
        self.first_tick = True

    def place_bo(self, qty, price):
        # Pre defined params
        # squareoff = 4 renko box, stoploss = 2 renko box, trailing stoploss = 1 renko box
        squareoff = round((self.get_brick_size() * 4),2)
        stoploss = round((self.get_brick_size() * 2),2)
        trailing_stoploss = round(self.get_brick_size(),2)
        if self.long:
            self.bo_buy(qty, price, squareoff, stoploss, trailing_stoploss)
        elif self.short:
            self.bo_sell(qty, price, squareoff, stoploss, trailing_stoploss)

    def bo_buy(self, qty, price, squareoff, stoploss, trailing_stoploss):
        # Here define the values of squareoff , stoploss, trailing_stoploss
        self.price = price
        self.squareoff = price + squareoff
        self.stoploss = price - stoploss
        self.trailing_stoploss = trailing_stoploss

        # self.print_w("Buy Placed: Price:{0}, Squareoff:{1}, Stoploss:{2}, trailing_stoploss:{3}".format(self.price,self.squareoff,self.stoploss,self.trailing_stoploss))

    def bo_sell(self, qty, price, squareoff, stoploss, trailing_stoploss):
        # Here define the values of squareoff , stoploss, trailing_stoploss
        self.price = price
        self.squareoff = price - squareoff
        self.stoploss = price + stoploss
        self.trailing_stoploss = trailing_stoploss

        # self.print_w("Sell Placed: Price:{0}, Squareoff:{1}, Stoploss:{2}, trailing_stoploss:{3}".format(self.price,self.squareoff,self.stoploss,self.trailing_stoploss))

    def update_bo(self, timestamp, ltp, no_of_box, dir):
        # Here according to ltp we will check if a bo has been satisfied
        # Then we will execute that bo
        if self.long:
            # if long check for the following
            if ltp <= self.stoploss or ltp >= self.squareoff:
                # exit the order
                self.print_w("TimeStamp: {0} \tExit Long\t LTP: {1}".format(timestamp, ltp))
                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, ltp, self.get_profit_loss(float(ltp))))
                self.pnl += self.get_profit_loss(float(ltp))
                self.long = False
                self.pos = 0.0
                self.unrealized_pnl = 0.0
                return 1

            if no_of_box != 0 and dir > 0:
                # if the stock moves by the amt in trailing_stoploss then increase the stoploss
                # by that amt
                self.stoploss += round(self.trailing_stoploss * no_of_box,2)
                # self.print_w("\tStoploss Moved!: {0}".format(self.stoploss))

        elif self.short:
            # if short check for the following
            if ltp >= self.stoploss or ltp <= self.squareoff:
                # exit the order
                self.print_w("TimeStamp: {0} \tExit Short\t LTP: {1}".format(timestamp, ltp))
                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, ltp, self.get_profit_loss(float(ltp))))
                self.pnl += self.get_profit_loss(float(ltp))
                self.short = False
                self.pos = 0.0
                self.unrealized_pnl = 0.0
                return 1

            if no_of_box != 0 and dir < 0:
                # if the stock moves by the amt in trailing_stoploss then increase the stoploss
                # by that amt
                self.stoploss += round(self.trailing_stoploss * no_of_box,2)
                # self.print_w("\tStoploss Moved!: {0}".format(self.stoploss))

        return 0

    def get_profit_loss(self, ltp):
        if self.long:
            return self.pos_qty * (ltp - self.pos)
        elif self.short:
            return self.pos_qty * (self.pos - ltp)

    def cal_change(self, ltp):
        if ltp > self.open:
            # Its a +ve change
            return round(((ltp - self.open) / self.open) * 100, 2)
        else:
            # Its a -ve change
            return round(((self.open - ltp) / self.open) * 100, 2)

    # Setting brick size. Auto mode is preferred, it uses history
    def set_brick_size(self, HLC_history=None, auto=True, brick_size=10.0):
        if auto == True:
            self.brick_size = self.__get_optimal_brick_size(HLC_history.iloc[:, [0, 1, 2]])
        else:
            self.brick_size = brick_size
        return self.brick_size

    def get_brick_size(self):
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
                self.renko_prices.append(
                    str(float(self.renko_prices[-1]) + 2 * float(self.brick_size) * int(np.sign(gap_div))))
                self.renko_directions.append(str(np.sign(gap_div)))
            # else:
            # num_new_bars = 0

            if is_new_brick:
                # Add each brick
                for d in range(start_brick, np.abs(gap_div)):
                    self.renko_prices.append(
                        str(float(self.renko_prices[-1]) + float(self.brick_size) * int(np.sign(gap_div))))
                    self.renko_directions.append(str(np.sign(gap_div)))

        return num_new_bars, self.renko_directions[-1]

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
        # get the mean of open close
        self.oc_2.clear()

        for i in range(1, len(self.renko_prices)):
            j = i - 1
            self.oc_2.append((float(self.renko_prices[j]) + float(self.renko_prices[i])) / 2.0)

        return np.array(self.oc_2)

    def get_qty(self):
        # Basicaly this func will load the scores of
        # the stock an decide on the quantity
        # Currently defaulted to 1
        return 1

    def get_close_price(self):
        self.close.clear()

        for i in range(len(self.renko_prices)):
            self.close.append(float(self.renko_prices[i]))

        return np.array(self.close)

    def get_hma(self, price, timeperiod=14):
        # HMA= WMA(2*WMA(n/2) − WMA(n)),sqrt(n))
        return (talib.WMA(
            2 * talib.WMA(price, timeperiod=math.floor(timeperiod / 2)) - talib.WMA(price, timeperiod=timeperiod),
            timeperiod=math.sqrt(timeperiod)))

    def get_rsi(self, price, timperiod=14):
        # return rsi
        return (talib.RSI(price, timperiod))

    def get_sma(self):
        # Calculate sma
        return talib.SMA(self.get_oc2_price(), timeperiod=8)

    def print_w(self, statement):
        order_dir_path = "G:\AI Trading\Code\RayTrader_v3\Backtest\orders_test_atr_bo\\" + self.stock_name + ".txt"
        with open(order_dir_path, "a") as text_file:
            print(statement, file=text_file)

    def on_tick(self, tick):
        # Once the tick is received now process it to decide
        # To Buy or Sell
        if self.first_tick:
            self.do_next(pd.Series([tick['last_price']]))
            self.open = float(tick['last_price'])
            self.cal_change(float(tick['last_price']))
            self.first_tick = False
            self.print_w("Saving first tick as open: {0}".format(self.open))

        else:
            new_renko_bars, dir = self.do_next(pd.Series([tick['last_price']]))

            if new_renko_bars != 0:
                self.unrealized_pnl = self.get_profit_loss(float(tick['last_price']))
                if self.unrealized_pnl is None:
                    self.unrealized_pnl = 0.0

                # Calculate % change
                self.cal_change(float(tick['last_price']))
                self.print_w("TimeStamp: {0} Number of Bars: {1} Unrealised PnL: {2} Change: {3}".format(tick['timestamp'],
                                                                                                         new_renko_bars,
                                                                                                         self.unrealized_pnl,
                                                                                                         self.change))

                # Update BO
                exe = self.update_bo(tick['timestamp'], float(tick['last_price']), new_renko_bars, int(dir))
                if exe == 1:
                    return

                # If in one tick the price jumped more than 4 then avoid
                if new_renko_bars < 4:
                    # Get hma
                    hma = float(self.get_hma(self.get_close_price())[-1])
                    # Get Rsi
                    rsi = float(self.get_rsi(self.get_close_price())[-1])
                    # now check if its +ve or -ve then check hma to decide buy or sell
                    if int(dir) > 0:
                        last_renko_price = float(self.renko_prices[-1])
                        # check if last_renko_price is above hma and not already bought
                        if hma <= last_renko_price and rsi > 50.5 and self.long == False:
                            # check if the stock is already shorted
                            if self.short:
                                self.print_w(
                                    "TimeStamp: {0} \tExit Short LTP: {1} RSI: {2} HMA: {3}".format(tick['timestamp'],
                                                                                                    tick['last_price']
                                                                                                    , round(rsi, 2),
                                                                                                    round(hma, 2)))
                                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, tick['last_price'],
                                                                                 self.get_profit_loss(
                                                                                     float(tick['last_price']))))
                                self.pnl += self.get_profit_loss(float(tick['last_price']))
                                self.short = False
                                self.pos = 0.0
                                self.unrealized_pnl = 0.0
                                # exit Short pos

                            self.print_w("TimeStamp: {0} \tBuy Stock\t LTP: {1} RSI: {2} HMA: {3}".format(tick['timestamp'],
                                                                                                          tick['last_price']
                                                                                                          , round(rsi, 2),
                                                                                                          round(hma, 2)))
                            self.pos = float(tick['last_price'])
                            self.long = True
                            self.place_bo(self.get_qty(), float(tick['last_price']))

                    else:
                        last_renko_price = float(self.renko_prices[-1])
                        # check if sma has crossed the renko bar
                        if last_renko_price <= hma and rsi < 49.5 and self.short == False:
                            # check if stock already long
                            if self.long:
                                self.print_w(
                                    "TimeStamp: {0} \tExit Long\t LTP: {1} RSI: {2} HMA: {3}".format(tick['timestamp'],
                                                                                                     tick['last_price']
                                                                                                     , round(rsi, 2),
                                                                                                     round(hma, 2)))
                                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, tick['last_price'],
                                                                                 self.get_profit_loss(
                                                                                     float(tick['last_price']))))
                                self.pnl += self.get_profit_loss(float(tick['last_price']))
                                self.long = False
                                self.pos = 0.0
                                self.unrealized_pnl = 0.0
                                # exit Long pos

                            self.print_w(
                                "TimeStamp: {0} \tSell Stock!\t LTP: {1} RSI: {2} HMA: {3}".format(tick['timestamp'],
                                                                                                   tick['last_price']
                                                                                                   , round(rsi, 2),
                                                                                                   round(hma, 2)))
                            self.pos = float(tick['last_price'])
                            self.short = True
                            self.place_bo(self.get_qty(), float(tick['last_price']))


# -------------------- Create and load Instances ------------------

historical_dir = "G:\AI Trading\Code\RayTrader_v3\Backtest\historical_data\\"
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

    return df


# Create class objects of all the stocks that are subscribed to
for f in stock_list:
    trader = RayTrader(f)
    hist_path = historical_dir + f + file_end
    df = csv_to_df(hist_path)
    trader.set_brick_size(auto = True, HLC_history = df[["High", "Low", "Close"]])
    trader.build_history(prices=df.Close)
    class_object_dict[f] = trader


# ------------------ Load live ticks ----------------------------


def on_ticks(ws, ticks):
    # Callback to receive ticks.
    # timestamp = datetime.now()
    # Now seperate the ticks and call on_tick of repective objects
    for tick in ticks:
        stock_symbol = stock_list[token_list.index(tick['instrument_token'])]
        trader_obj = class_object_dict[stock_symbol]
        trader_obj.on_tick(tick)
        # stock_file_path = dir_path + "/" + stock_symbol + ".csv"
        # df = pd.DataFrame(tick, index=[timestamp])
        # df.to_csv(stock_file_path, mode='a', header=False)


def start_trade(csv_file, stock):
    # Load the live tick file and call in the on_tick_func
    df = pd.read_csv(csv_file,
                     names=['timestamp', 'tradable', 'mode', 'instrument_token', 'last_price'])

    # df = df.drop(['timestamp'], axis=1)

    # parse date
    # df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    # Set index
    # df = df.set_index('timestamp')

    trader_obj = class_object_dict[stock.split(".")[0]]
    trader_obj.print_w("Stock: {0}".format(os.path.split(stock)[-1]))
    trader_obj.print_w("Optimal Renko size: {0}".format(trader_obj.get_brick_size()))

    for i in range(len(df)):
        trader_obj.on_tick(df.iloc[i])

    if trader_obj.unrealized_pnl != 0.0:
        trader_obj.pnl += trader_obj.unrealized_pnl
        trader_obj.unrealized_pnl = 0.0

    trader_obj.print_w("Final PnL: {0}".format(trader_obj.pnl))


# ----------------------- Main -------------------------------------------------

# Load the parent dir
# files = glob.glob("/home/ubuntu/zerodha/live_data/" + str(date.today()) + "/*.csv")
files = glob.glob("G:\AI Trading\Code\RayTrader_v3\Backtest\live_ticks\\2019-02-25\*.csv")

# threads_dict = {}

#Single Threaded
for f in files:
    start = time.time()
    start_trade(f, os.path.split(f)[-1])
    end = time.time()
    print("Stock: {0} Time req: {1}".format(os.path.split(f)[-1],(end-start)))

#Multi-threaded
#Create all threads
# print("Multi-threaded")
# for f in files:
#     stock_name = os.path.split(f)[-1]
#     threads_dict[stock_name.split(".")[0]] = threading.Thread(target=start_trade, args=(f,stock_name,))
#
# #start all threads
# for t in threads_dict.values():
#     t.start()
#
# #Join all threads
# for t in threads_dict.values():
#     t.join()

#Never use this
# if __name__ == "__main__":
#     #Multi-Processing
#     print("Multi-Processing")
#     for f in files:
#         stock_name = os.path.split(f)[-1]
#         threads_dict[stock_name.split(".")[0]] = multiprocessing.Process(target=start_trade, args=(f,stock_name,))
#
#     #start all threads
#     for t in threads_dict.values():
#         t.start()
#
#     #Join all threads
#     for t in threads_dict.values():
#         t.join()