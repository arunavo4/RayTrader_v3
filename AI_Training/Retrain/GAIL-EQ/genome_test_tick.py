"""
################## Single Stock Live Tick Trader #####################
# Real World Simulation
# Live test of a Single Stock 

# Trader Based on the Neat AI 
# Takes in the Indicators and patters.
# Outputs Buy || Sell || Hold

"""

import pickle
import os, math
import glob
import numpy as np
import pandas as pd
import talib
import time
import threading
import multiprocessing
from datetime import datetime, date


# -------------------------------------------------------------

# Genome list 
genomes = glob.glob(".\\genomes\\*.pkl")
genome_list = [ os.path.split(g)[-1] for g in genomes ]

# Create a dir to store today's order data
dir_path = ".\orders\\" + str(date.today()) + "\\"

if not os.path.exists(dir_path):
    try:
        os.makedirs(dir_path, exist_ok=True)  # succeeds even if directory exists.
    except FileExistsError:
        print("Dir creation failed!")
        pass


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
        self.interval = 1
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

    def __init__(self, id):
        self.genome_id = id
        self.tradable = False
        self.open_trade = False
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
        self.balance = 2000
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
        # #Hack to make it work
        # #Append two more same data
        # self.df_stock = self.df_stock.append(price_df)
        # self.df_stock = self.df_stock.append(price_df)


    def load_hist_data(self, data):
        self.df_stock = data

    def set_model_path(self, path):
        self.model_path = path

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
                self.check_tradable()
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
                self.check_tradable()
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
        self.balance += self.pnl

    def get_profit_loss(self, ltp):
        if self.long:
            return round((self.pos_qty * (ltp - self.pos)),2)
        elif self.short:
            return round((self.pos_qty * (self.pos - ltp)),2)


    def cal_change(self, ltp):
        if ltp > self.open:
            # Its a +ve change
            self.change = round(((ltp - self.open) / self.open) * 100, 2)
        else:
            # Its a -ve change
            self.change = round(((self.open - ltp) / self.open) * 100, 2)

    def get_qty(self, price):
        # Depending on the number of stocks
        # equally divide the amount
        amount = float(self.balance) * self.leverage
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
                self.check_tradable()
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
                self.check_tradable()
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
        order_dir_path = dir_path + self.genome_id + ".txt"
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
historical_dir = "G:\AI Trading\Code\RayTrader_v3\Backtest\historical_data\\"
file_end = "-EQ.csv"


# For Minute Data
# Create class objects of all the stocks that are subscribed to
for g in genome_list:
    obj = MinuteData(g)
    class_min_dict[g] = obj


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
for g in genome_list:
    trader = RayTrader(g)
    hist_path = historical_dir + f + file_end
    df = csv_to_df(hist_path)
    class_object_dict[f] = trader


# ------------------ Load live ticks ----------------------------


def start_trade(csv_file, stock):
    # Load the live tick file and call in the on_tick_func
    df = pd.read_csv(csv_file,
                     names=['timestamp', 'tradable', 'mode', 'instrument_token', 'last_price'])

    # df = df.drop(['timestamp'], axis=1)

    # parse date
    # df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')
    # Set index
    # df = df.set_index('timestamp')
    
    for i in range(len(df)):
        for g in genome_list:
            trader_obj = class_object_dict[g]
            trader_obj.on_tick(df.iloc[i])

    # if trader_obj.unrealized_pnl != 0.0:
    #     trader_obj.pnl += trader_obj.unrealized_pnl
    #     trader_obj.unrealized_pnl = 0.0

    # trader_obj.print_w("Final PnL: {0}".format(trader_obj.pnl))


# ----------------------- Main -------------------------------------------------

# Load the parent dir
# files = glob.glob("/home/ubuntu/zerodha/live_data/" + str(date.today()) + "/*.csv")
files = glob.glob("G:\\AI Trading\\Code\\RayTrader_v3\\Backtest\\Live_BackTest\\live_ticks\\*\\ADANIPORTS.csv")

# threads_dict = {}

# Single Threaded
for f in files:
    print("Processing: ", os.path.split(f)[-1])
    start = time.time()
    start_trade(f, os.path.split(f)[-1])
    end = time.time()
    print("Stock: {0} Time req: {1}".format(os.path.split(f)[-1],(end-start)))
    break
