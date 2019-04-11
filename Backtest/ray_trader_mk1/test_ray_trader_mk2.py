"""
#  Trader to trade based on time series forecasting.
"""

import configparser, pickle
import time, os, sys
import warnings
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import logging
import glob
# import talib
import threading
import tensorflow as tf
import numpy as np
from datetime import datetime, date
# from kiteconnect import KiteConnect, KiteTicker
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
tf.logging.info('TensorFlow')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')

# Config file
# config = configparser.ConfigParser()
# config.read('/home/ubuntu/zerodha/config.cfg')

# Initialize from config file
# API_KEY = config['KITE']['API_KEY']
# API_SECRET = config['KITE']['API_SECRET']
# ACCESS_TOKEN = config['LOGIN']['ACCESS_TOKEN']
# PUBLIC_TOKEN = config['LOGIN']['PUBLIC_TOKEN']
# BALANCE = config['PROFILE']['BALANCE']

BALANCE = 2000
init_bal = BALANCE
target_percent = 15
stoploss_percent = 7.5
stop_trading = False

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# Init KiteConnect
# kite = KiteConnect(api_key=API_KEY)

# Set Access Token
# kite.set_access_token(ACCESS_TOKEN)

# Read the List of stocks to subscribe to
with open('.\symbol_token.out', 'rb') as fp:
    stock_token_dict = pickle.load(fp)

# Seperate Symbols and Tokens
stock_list = list(stock_token_dict.keys())

token_list = list(stock_token_dict.values())

token_list = [int(token) for token in token_list]

# Create a dir to store today's order data
dir_path = ".\orders\\" + str(date.today()) + "\\"

if not os.path.exists(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)  # succeeds even if directory exists.
    except FileExistsError:
        print("Dir creation failed!")
        pass

# Read the Optimal Renko Box size
with open('.\opt_renko_box.out', 'rb') as f:
    opt_renko_box = pickle.load(f)  # use `pickle.loads` to do the reverse

# Dict to store class objects to reference later
class_object_dict = {}

# Dict to store class objects of MinuteData
class_min_dict = {}

# list to store current tradable stocks
selected_stocks = []


# Class to deal with converting tick data to 15 min ohlc data
class MinuteData:

    def __init__(self, stock):
        self.stock_name = stock
        self.open = 0.0
        self.high = 0.0
        self.low = 0.0
        self.close = 0.0
        self.vol = 0
        self.minute = -1
        self.interval = 15
        self.last_price = []

    def get_last_price(self):
        return self.last_price

    def re_initialize(self, timestamp, ltp):
        self.open = ltp
        self.high = ltp
        self.low = ltp
        self.close = ltp
        self.minute = timestamp.minute

    def update(self, timestamp, ltp):
        # First time and Check the time
        if self.minute == -1:
            self.re_initialize(timestamp, ltp)

        elif self.minute != timestamp.minute and (timestamp.minute % self.interval) == 0:
            # Not the first time
            if self.minute != -1:
                # write data to file
                self.save_data(timestamp)
            self.re_initialize(timestamp, ltp)

        # check ltp decide high low
        if ltp < self.low:
            self.low = ltp
        if ltp > self.high:
            self.high = ltp

        self.close = ltp

    def save_data(self, timestamp):
        # Form the line : 22-02-2019 03:29:00 PM,354.6,354.6,354,354.05,4344
        timestamp = timestamp.replace(minute=self.minute, second=0, microsecond=0)
        line = [timestamp.strftime('%d-%m-%Y %I:%M:%S %p'), self.open, self.high, self.low, self.close]

        self.last_price = line
        self.print_w("{}".format(line))

    def print_w(self, statement):
        order_dir_path = dir_path + self.stock_name + ".txt"
        with open(order_dir_path, "a") as text_file:
            print(statement, file=text_file)

# Trader class to handle all trading funcs
class RayTrader:

    def __init__(self, stock):
        self.stock_name = stock
        self.tradable = False
        self.open_trade = False
        self.valid_set_size_percentage = 10
        self.test_set_size_percentage = 10
        self.seq_len = 20
        self.leverage = 15
        self.target = 0.0
        self.loss = 0.0
        self.orders = []
        self.open = 0.0
        self.change = 0.0
        self.current_price = []
        self.predicted_price = []
        self.last_price = 0.0
        self.close = []
        self.long = False
        self.short = False
        self.pos = 0.0
        self.trail_pos = 0.0
        self.pos_qty = 1
        self.pnl = 0.0
        self.unrealized_pnl = 0.0
        self.first_tick = True

    def add_hist_data(self, price_list):
        #Hack to make it work
        #Remove two duplicate rows
        self.df_stock = self.df_stock.iloc[:-2]
        #add the new data
        price_df = pd.DataFrame(data=[[price_list[0], price_list[1], price_list[2], price_list[3], price_list[4]]],
                                columns=['Date', 'Open', 'High', 'Low', 'Close'])
        # parse date
        price_df['Date'] = pd.to_datetime(price_df['Date'], format='%d-%m-%Y %I:%M:%S %p')
        # Set index
        price_df = price_df.set_index('Date')
        #append
        self.df_stock = self.df_stock.append(price_df)
        #Hack to make it work
        #Append two more same data
        self.df_stock = self.df_stock.append(price_df)
        self.df_stock = self.df_stock.append(price_df)


    def load_hist_data(self, data):
        self.df_stock = data

    def set_model_path(self, path):
        self.model_path = path

    def load_data(self, stock, seq_len):
        # data_raw = stock.as_matrix()
        data_raw = stock.values
        data = []
        for index in range(len(data_raw) - seq_len):
            data.append(data_raw[index: index + seq_len])
        data = np.array(data)
        valid_set_size = int(np.round(self.valid_set_size_percentage / 100 * data.shape[0]))
        test_set_size = int(np.round(self.test_set_size_percentage / 100 * data.shape[0]))
        train_set_size = data.shape[0] - (valid_set_size + test_set_size)
        x_train = data[:train_set_size, :-1, :]
        y_train = data[:train_set_size, -1, :]
        x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
        y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]
        x_test = data[train_set_size + valid_set_size:, :-1, :]
        y_test = data[train_set_size + valid_set_size:, -1, :]
        return [x_train, y_train, x_valid, y_valid, x_test, y_test]

    def normalize_data(self, df):
        self.min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        df['Open'] = self.min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
        df['High'] = self.min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
        df['Low'] = self.min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
        df['Close'] = self.min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
        return df

    def predict_price(self, price_list, first_time=False):
        # add the price_list to hist_data
        if not first_time:
            self.current_price = price_list[1:]
            self.add_hist_data(price_list)

        #print the last few of data frame
        self.print_w("Dataframe tail : {}".format(self.df_stock.tail()))
        # norm the data
        df_stock_norm = self.df_stock.copy()
        df_stock_norm = self.normalize_data(df_stock_norm)

        # load the tensorflow graph and use it to predict
        x_train, y_train, x_valid, y_valid, x_test, y_test = self.load_data(df_stock_norm, self.seq_len)

        tf.reset_default_graph()

        # Fitting the model
        with tf.Session(graph=tf.Graph()) as sess:
            tf.saved_model.loader.load(sess, ["serve"], self.model_path)
            graph = tf.get_default_graph()
            # print(graph.get_operations())
            X_graph = graph.get_tensor_by_name('myInput:0')
            output_graph = graph.get_tensor_by_name('myOutput:0')
            # Predictions
            y_test_pred = sess.run(output_graph,
                                   feed_dict={X_graph: x_test})

        # scale back the values
        x_test = self.min_max_scaler.inverse_transform(x_test[:,18,3].reshape(-1, 1))
        y_test = self.min_max_scaler.inverse_transform(y_test)
        y_test_pred = self.min_max_scaler.inverse_transform(y_test_pred)

        self.predicted_price = y_test_pred[-2:,3]
        self.print_w("Input price ohlc: {}".format(x_test[-5:]))
        self.print_w("Actual price ohlc: {}".format(y_test[-5:,3]))
        self.print_w("Predicted price ohlc: {}".format(self.predicted_price))
        # return the last y predicted value
        return y_test_pred[-2:,3]

    def set_target_loss(self, price):
        # currently 1% target 0.5% stoploss
        self.target = (price * 0.01)
        self.loss = (price * 0.005)

    def check_tradable(self):
        # Here basically change the status
        # Check if a stock has hit target or stoploss
        if self.pnl <= (self.loss * self.pos_qty):
            self.print_w("Stoploss hit for the day stopping Trade")
            self.tradable = False
        elif self.pnl >= (self.target * self.pos_qty):
            self.print_w("Target reached Stop for the day")
            self.tradable = False

    def place_bo(self, qty, price):
        # Pre defined params
        # squareoff = 0.6%, stoploss = 0.3%, trailing stoploss = 0.2%
        if qty >= 1:
            squareoff = round((price*0.006), 2)
            stoploss = round((price*0.003), 2)
            trailing_stoploss = round((price*0.002), 2)

            if self.long:
                self.bo_buy(qty, price, squareoff, stoploss, trailing_stoploss)
                self.open_trade = True
                # self.bo_order_buy(self.stock_name,1,price=price,squareoff=squareoff,stoploss=stoploss,trailing_stoploss=trailing_stoploss)
                self.print_w(
                    "Buy Order: qty:{0}, price:{1}, squareoff:{2}, stoploss:{3}, trailing_stoploss:{4}".format(qty,
                                                                                                               price,
                                                                                                               squareoff,
                                                                                                               stoploss,
                                                                                                               trailing_stoploss))
            elif self.short:
                self.bo_sell(qty, price, squareoff, stoploss, trailing_stoploss)
                self.open_trade = True
                # self.bo_order_sell(self.stock_name,1,price=price,squareoff=squareoff,stoploss=stoploss,trailing_stoploss=trailing_stoploss)
                self.print_w(
                    "Sell Order: qty:{0}, price:{1}, squareoff:{2}, stoploss:{3}, trailing_stoploss:{4}".format(qty,
                                                                                                                price,
                                                                                                                squareoff,
                                                                                                                stoploss,
                                                                                                                trailing_stoploss))
        else:
            self.print_w("Not Enough Funds!")

    def bo_buy(self, qty, price, squareoff, stoploss, trailing_stoploss):
        # Here define the values of squareoff , stoploss, trailing_stoploss
        self.price = price
        self.squareoff = round((price + squareoff),2)
        self.stoploss = round((price - stoploss),2)
        self.trailing_stoploss = round(trailing_stoploss,2)
        self.print_w(
            "Buy Order: qty:{0}, price:{1}, squareoff:{2}, stoploss:{3}, trailing_stoploss:{4}".format(qty,
                                                                                                       self.price,
                                                                                                       self.squareoff,
                                                                                                       self.stoploss,
                                                                                                       self.trailing_stoploss))

    def bo_sell(self, qty, price, squareoff, stoploss, trailing_stoploss):
        # Here define the values of squareoff , stoploss, trailing_stoploss
        self.price = price
        self.squareoff = round((price - squareoff),2)
        self.stoploss = round((price + stoploss),2)
        self.trailing_stoploss = round(trailing_stoploss,2)
        self.print_w(
            "Sell Order: qty:{0}, price:{1}, squareoff:{2}, stoploss:{3}, trailing_stoploss:{4}".format(qty,
                                                                                                        self.price,
                                                                                                        self.squareoff,
                                                                                                        self.stoploss,
                                                                                                        self.trailing_stoploss))

    def update_bo(self, timestamp, ltp):
        # Here according to ltp we will check if a bo has been satisfied
        # Then we will execute that bo
        if self.long:
            # if long check for the following
            if ltp <= self.stoploss or ltp >= self.squareoff:
                # exit the order
                self.print_w("Stoploss / Squareoff hit!")
                self.print_w("TimeStamp: {0} \tExit Long\t LTP: {1}".format(timestamp, ltp))
                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, ltp, self.get_profit_loss(float(ltp))))
                self.pnl += self.get_profit_loss(float(ltp))
                self.long = False
                self.pos = self.trail_pos = 0.0
                self.unrealized_pnl = 0.0
                self.open_trade = False
                self.update_balance()
                print("Exit Long {0} Pnl: {1}".format(self.stock_name, self.pnl))
                check_traget_stoploss()
                # self.check_tradable()
                return 1

            if float(ltp) > self.pos:
                no_of_boxes = math.floor((float(ltp) - self.trail_pos) / self.trailing_stoploss)
                # if the stock moves by the amt in trailing_stoploss then increase the stoploss
                # by that amt
                if no_of_boxes >= 1:
                    self.stoploss += self.trailing_stoploss * no_of_boxes
                    self.trail_pos = float(ltp)
                    self.print_w("Stoploss moved: {}".format(self.stoploss))

        elif self.short:
            # if short check for the following
            if ltp >= self.stoploss or ltp <= self.squareoff:
                # exit the order
                self.print_w("Stoploss / Squareoff hit!")
                self.print_w("TimeStamp: {0} \tExit Short\t LTP: {1}".format(timestamp, ltp))
                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, ltp, self.get_profit_loss(float(ltp))))
                self.pnl += self.get_profit_loss(float(ltp))
                self.short = False
                self.pos = self.trail_pos = 0.0
                self.unrealized_pnl = 0.0
                self.open_trade = False
                self.update_balance()
                print("Exit Short {0} Pnl: {1}".format(self.stock_name, self.pnl))
                check_traget_stoploss()
                # self.check_tradable()
                return 1

            if float(ltp) < self.pos:
                no_of_boxes = math.floor((self.trail_pos - float(ltp)) / self.trailing_stoploss)
                # if the stock moves by the amt in trailing_stoploss then increase the stoploss
                # by that amt
                if no_of_boxes >= 1:
                    self.stoploss -= self.trailing_stoploss * no_of_boxes
                    self.trail_pos = float(ltp)
                    self.print_w("Stoploss moved: {}".format(self.stoploss))

        return 0

    def update_balance(self):
        global BALANCE
        BALANCE += self.pnl

    def get_profit_loss(self, ltp):
        if self.long:
            return round((self.pos_qty * (ltp - self.pos)),2)
        elif self.short:
            return round((self.pos_qty * (self.pos - ltp)),2)

    def cal_predicted_change(self, ltp):
        #predicted change will be based on the 
        #predicted price previous and last
        prev_predicted_close = self.predicted_price[0]
        predicted_close = self.predicted_price[1]
        change = 0.0

        # cal change in the close price
        if prev_predicted_close < predicted_close:
            # predicted -ve change
            change = round(((predicted_close - prev_predicted_close) / prev_predicted_close) * 100, 2)
        else:
            change = round(((prev_predicted_close - predicted_close) / prev_predicted_close) * 100, 2)

        self.print_w("Predicted change: {}".format(change))
        return change


    def cal_change(self, ltp):
        if ltp > self.open:
            # Its a +ve change
            self.change = round(((ltp - self.open) / self.open) * 100, 2)
        else:
            # Its a -ve change
            self.change = round(((self.open - ltp) / self.open) * 100, 2)

    def get_tradable_stocks(self):
        count = 0
        for s in selected_stocks:
            stock_symbol = stock_list[token_list.index(s[0])]
            trader = class_object_dict[stock_symbol]
            if trader.tradable:
                count += 1
        return count

    def get_qty(self, price):
        # Depending on the number of stocks
        # equally divide the amount
        tradable_stock_count = class_object_dict
        amount = (float(BALANCE) / float(self.get_tradable_stocks())) * self.leverage
        self.pos_qty = math.floor(amount / price)
        return self.pos_qty

    def exit_trades(self, tick, timestamp):
        # if there are any open trades exit them
        if self.open_trade:
            # exit trade
            if self.short:
                self.print_w("TimeStamp: {0} \tExit Short LTP: {1}".format(timestamp, tick['last_price']))
                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, tick['last_price'],
                                                                 self.get_profit_loss(float(tick['last_price']))))
                self.pnl += self.get_profit_loss(float(tick['last_price']))
                self.short = False
                self.pos = 0.0
                self.unrealized_pnl = 0.0
                self.update_balance()
                print("Exit Short {0} Pnl: {1}".format(self.stock_name, self.pnl))
                check_traget_stoploss()
                # self.check_tradable()
                # exit Short pos

            # check if stock already long
            if self.long:
                self.print_w("TimeStamp: {0} \tExit Long\t LTP: {1}".format(timestamp, tick['last_price']))
                self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, tick['last_price'],
                                                                 self.get_profit_loss(float(tick['last_price']))))
                self.pnl += self.get_profit_loss(float(tick['last_price']))
                self.long = False
                self.pos = 0.0
                self.unrealized_pnl = 0.0
                self.update_balance()
                print("Exit Long {0} Pnl: {1}".format(self.stock_name, self.pnl))
                check_traget_stoploss()
                # self.check_tradable()
                # exit Long pos

    # def bo_order_buy(self, stock_symbol, qty, price, squareoff, stoploss, trailing_stoploss):
    #     # Place a BO order for BUY
    #     try:
    #         order_id = kite.place_order(tradingsymbol=stock_symbol,
    #                                     exchange=kite.EXCHANGE_NSE,
    #                                     transaction_type=kite.TRANSACTION_TYPE_BUY,
    #                                     order_type=kite.ORDER_TYPE_LIMIT,
    #                                     quantity=qty,
    #                                     product=kite.PRODUCT_MIS,
    #                                     price=price,
    #                                     variety=kite.VARIETY_BO,
    #                                     squareoff=squareoff,
    #                                     stoploss=stoploss,
    #                                     trailing_stoploss=trailing_stoploss)

    #         logging.info("Order placed. ID is: {}".format(order_id))

    #         return order_id
    #     except Exception as e:
    #         logging.info("Order placement failed: {}".format(e.message))

    # def bo_order_sell(self, stock_symbol, qty, price, squareoff, stoploss, trailing_stoploss):
    #     # Place a BO order SELL
    #     try:
    #         order_id = kite.place_order(tradingsymbol=stock_symbol,
    #                                     exchange=kite.EXCHANGE_NSE,
    #                                     transaction_type=kite.TRANSACTION_TYPE_SELL,
    #                                     order_type=kite.ORDER_TYPE_LIMIT,
    #                                     quantity=qty,
    #                                     product=kite.PRODUCT_MIS,
    #                                     price=price,
    #                                     variety=kite.VARIETY_BO,
    #                                     squareoff=squareoff,
    #                                     stoploss=stoploss,
    #                                     trailing_stoploss=trailing_stoploss)

    #         logging.info("Order placed. ID is: {}".format(order_id))

    #         return order_id
    #     except Exception as e:
    #         logging.info("Order placement failed: {}".format(e.message))

    # def co_order(self, stock_symbol, price):
    #     # Place a CO order
    #     try:
    #         order_id = kite.place_order(tradingsymbol=stock_symbol,
    #                                     exchange=kite.EXCHANGE_NSE,
    #                                     transaction_type=kite.TRANSACTION_TYPE_BUY,
    #                                     order_type=kite.ORDER_TYPE_LIMIT,
    #                                     quantity=1,
    #                                     product=kite.PRODUCT_MIS,
    #                                     price=price,
    #                                     variety=kite.VARIETY_CO,
    #                                     trigger_price=(self.get_brick_size()*2))

    #         logging.info("Order placed. ID is: {}".format(order_id))

    #         return order_id
    #     except Exception as e:
    #         logging.info("Order placement failed: {}".format(e.message))

    def print_w(self, statement):
        order_dir_path = dir_path + self.stock_name + ".txt"
        with open(order_dir_path, "a") as text_file:
            print(statement, file=text_file)

    def on_tick(self, tick, timestamp):
        # Once the tick is received now process it to decide to Buy or Sell
        # check time -->  if ear;ier than 9:15 store as open price
        # self.print_w("Stock : {0} Qty: {1}".format(self.stock_name,self.get_qty(float(tick['last_price']))))
        if self.first_tick:
            # self.do_next(pd.Series([tick['last_price']]))
            self.last_price = float(tick['last_price'])
            self.open = float(tick['last_price'])
            self.cal_change(float(tick['last_price']))
            self.set_target_loss(float(tick['last_price']))
            self.first_tick = False
            self.print_w("Saving first tick as open: {0}".format(self.open))

        else:

            # If only last_price has changed
            if self.last_price != float(tick['last_price']):

                self.last_price = float(tick['last_price'])
                self.unrealized_pnl = self.get_profit_loss(float(tick['last_price']))
                if self.unrealized_pnl is None:
                    self.unrealized_pnl = 0.0

                # Calculate % change
                self.cal_change(float(tick['last_price']))
                self.print_w("TimeStamp: {0} Ltp: {1} Unrealised PnL: {2} Change: {3}".format(timestamp, float(tick['last_price']),
                                                                                              round(self.unrealized_pnl,2), self.change))

                # Update BO also check if order got executed --> then dont do anything in this tick
                exe = self.update_bo(timestamp, float(tick['last_price']))
                if exe == 1:
                    return

                # now check if there is an already open trade
                if not self.open_trade:
                    # Now decide whether to buy or sell stocks
                    ltp = float(tick['last_price'])
                    if self.predicted_price[0] < self.predicted_price[1]:
                        # check if stock not already bought
                        if self.long == False:
                            self.pos = self.trail_pos = float(tick['last_price'])
                            self.long = True
                            self.place_bo(self.get_qty(float(tick['last_price'])), float(tick['last_price']))
                            self.print_w(
                                "TimeStamp: {0} \tBuy Stock\t Qty: {1} LTP: {2}".format(timestamp, self.pos_qty,
                                                                                        tick['last_price']))
                            print("TimeStamp: {0} \tBuy Stock:{1}\t Qty: {2} LTP: {3}".format(timestamp, self.stock_name,
                                                                                                 self.pos_qty,tick['last_price']))

                    else:
                        # check if stock not already sold
                        if self.short == False:
                            self.pos = self.trail_pos = float(tick['last_price'])
                            self.short = True
                            self.place_bo(self.get_qty(float(tick['last_price'])), float(tick['last_price']))
                            self.print_w(
                                "TimeStamp: {0} \tSell Stock!\t Qty: {1} LTP: {2}".format(timestamp, self.pos_qty,
                                                                                          tick['last_price']))
                            print("TimeStamp: {0} \tSell Stock:{1}\t Qty: {2} LTP: {3}".format(timestamp, self.stock_name,
                                                                                                 self.pos_qty,tick['last_price']))


# -------------------- Create and load Instances of Min Data ------------------
# For Minute Data
# Create class objects of all the stocks that are subscribed to
for f in stock_list:
    obj = MinuteData(f)
    class_min_dict[f] = obj


# -------------------- Stock Selection ---------------------------

# function to return the second element of the
# two elements passed as the paramater
def sortSecond(val):
    return val[1]


def check_traget_stoploss():
    global target_percent, stoploss_percent, BALANCE, init_bal, stop_trading

    pnl = int(BALANCE - init_bal)
    target = round(init_bal + (init_bal*(target_percent/100)),2)
    stoploss  = round(init_bal - (init_bal*(stoploss_percent/100)),2)
    if BALANCE >= target:
        print("Target Achieved for the Day!")
        stop_trading = True

    elif BALANCE <= stoploss:
        print("Stoploss hit for the Day!")
        stop_trading = True


# Func to predict the stock price and save it
def predict_stocks(ticks):
    global prediction_complete
    # go through the traders and call the predict func
    # also mark stocks non tradable
    for stock_symbol in stock_list:
        trader_obj = class_object_dict[stock_symbol]
        min_obj = class_min_dict[stock_symbol]
        trader_obj.predict_price(min_obj.get_last_price())
        trader_obj.tradable = False
    prediction_complete = True


# Func to select stocks and mark tradable
def select_stocks(ticks, return_no=1):
    global selected_stocks
    # list to store (instrument token, change) tuple

    #Here Check for overall target and stoploss
    #If that is hit then stop trading for the day
    check_traget_stoploss()

    if not stop_trading:
        # Here call all the traders and predict the next 15 min
        # and then I can sort out the amount of change and trade
        # accordingly while placing order give a 10% buffer
        stock_change = []
        # Now seperate the ticks
        for tick in ticks:
            # Store the rest of the change and sort it out
            stock_symbol = stock_list[token_list.index(tick['instrument_token'])]
            trader_obj = class_object_dict[stock_symbol]
            # select stocks which have a change of more than 0.6% atleast
            change = round(trader_obj.cal_predicted_change(float(tick['last_price'])), 2)
            if change > 0.6 and change < 5:
                tup = (int(tick['instrument_token']), change)
                stock_change.append(tup)

        # if +ve then sort stock with reverse=True and use change as key
        stock_change.sort(key=sortSecond, reverse=True)
        # Now mark those stocks tradable
        selected_stocks = stock_change[0:return_no]

        print("Selected Stocks: ",selected_stocks)

        for stock in selected_stocks:
            stock_token = stock[0]
            stock_symbol = stock_list[token_list.index(stock_token)]
            trader_obj = class_object_dict[stock_symbol]
            trader_obj.tradable = True


# -------------------- Create and load Instances ------------------

model_dir = "G:\AI Trading\Code\RayTrader_v3\Trader\model\\"
file_end_model = "-EQ"

historical_dir = ".\hist_15_data\\"
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
    # Drop 200 rows
    df = df.iloc[:-200]
    #The hack for it to take the last value
    #Add two rows for the perfect prediction
    df = df.append(df.iloc[-1])
    df = df.append(df.iloc[-1])
    # Drop the Vol
    df = df[['Open', 'High', 'Low', 'Close']]
    return df


# Create class objects of all the stocks that are subscribed to
for f in stock_list:
    trader = RayTrader(f)
    trader.print_w("Stock: {0}".format(os.path.split(f)[-1]))
    hist_path = historical_dir + f + file_end
    df = csv_to_df(hist_path)
    trader.load_hist_data(df)
    model_path = model_dir + f + file_end_model
    trader.set_model_path(model_path)
    trader.predict_price([], True)
    class_object_dict[f] = trader

# ------------------ Market Watch ----------------------------

# variable to run startup code
startup_code = True
close_code = True
prediction_complete = True
minute = -1
run_15_min_job = False
# Initialise KiteTicker -> Live Ticks
# kws = KiteTicker(API_KEY, ACCESS_TOKEN)

def on_ticks(ws, ticks, timestamp):
    global startup_code, close_code, prediction_complete, minute, run_15_min_job
    # Callback to receive ticks.
    # timestamp = datetime.now()

    # First time send the ticks to select the stocks that will be marked tradable
    if startup_code:
        if not prediction_complete:
            return
        print("Running Startup Code")
        select_stocks(ticks)
        startup_code = False

    else:
        if minute != timestamp.minute and (timestamp.minute % 15) == 0:
            run_15_min_job = True
            minute = timestamp.minute

        # Zerodha squareoff time is 3:20
        # So dont take any trades after 3 pm
        if timestamp.hour < 15:
            # Now seperate the ticks and call on_tick of repective objects
            for tick in ticks:
                stock_symbol = stock_list[token_list.index(tick['instrument_token'])]
                trader_obj = class_object_dict[stock_symbol]
                if trader_obj.tradable:
                    trader_obj.on_tick(tick, timestamp)
                    if run_15_min_job:
                        # exit from the trade
                        trader_obj.exit_trades(tick, timestamp)

                class_min_dict[stock_symbol].update(timestamp, float(tick['last_price']))

            if run_15_min_job:
                #Here you wanna call all the save func even if they
                #dont have tick this time
                for stock in stock_list:
                    if class_min_dict[stock].minute != timestamp.minute:
                        class_min_dict[stock].save_data(timestamp)
                        # class_min_dict[stock_symbol].re_initialize(timestamp, float(tick['last_price']))

                # stock_file_path = dir_path + "/" + stock_symbol + ".csv"
                # df = pd.DataFrame(tick, index=[timestamp])
                # df.to_csv(stock_file_path, mode='a', header=False)

        # Every 15 min interval call the predict func on a
        # seperate thread and dont wait, also stop trading
        if run_15_min_job:
            prediction_complete = False
            startup_code = True
            # create a thread to call the predict func
            print("Predict Stocks")
            predict_stocks(ticks)
            # t = threading.Thread(target=predict_stocks, args=(1,))
            # t.daemon = True
            # # start all threads dont join
            # t.start()
            # t.join()
            run_15_min_job = False

        # Stop the websocket after 3:30 pm
        if timestamp.hour == 15 and timestamp.minute > 30:
            if close_code:
                # print the final PnL
                for trader in class_object_dict.values():
                    if trader.unrealized_pnl != 0.0:
                        trader.pnl += trader.unrealized_pnl
                        trader.unrealized_pnl = 0.0
                    trader.print_w("Final PnL: {0}".format(trader.pnl))
                # exit the program
                close_code = False


# def on_connect(ws, response):
#     # Callback on successful connect.
#     # Subscribe to a list of instrument_tokens {All stocks present in historical data}
#     ws.subscribe(token_list)

#     # Set Mode QUOTE
#     # ws.set_mode(ws.MODE_LTP, token_list)

# # def on_close(ws, code, reason):
# #     # On connection close stop the main loop
# #     # Reconnection will not happen after executing `ws.stop()`
# #     ws.stop()

# # Assign the callbacks.
# kws.on_ticks = on_ticks
# kws.on_connect = on_connect
# # kws.on_close = on_close

# # Infinite loop on the main thread. Nothing after this will run.
# # You have to use the pre-defined callbacks to manage subscriptions.
# kws.connect()


# Load the tick data and call on_tick()
print("Loading tick data")
data_dir = [ ".\\live_ticks\\2019-03-06\\" + str(i) + ".out" for i in range(2,377)]

print("Initial Balance: ",BALANCE)

for in_file in data_dir:
    print("Processing :", in_file)
    # unpickle each file and the feed to on_ticks()
    live_ticks = []
    # Read from the file the stock list
    with open(in_file, 'rb') as fp:
        live_ticks = pickle.load(fp)

    for ticks in live_ticks:
        if stop_trading:
            break
        timestamp = datetime.fromtimestamp(os.path.getmtime(in_file))
        on_ticks(0, ticks, timestamp)

    if stop_trading:
        break

print("Final Balance: ",BALANCE)
print("Overall Pnl: ",round(BALANCE-init_bal,2))