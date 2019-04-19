"""
#################################################################################
#   This trader implements RSI + HMA + Opt RENKO Strategy
#   along with stock selection and target and stoploss for
#   each trader.
#
#   Look into nifty direction at 9:30 and if +ve take top 6 gainers stocks
#   else top 6 loosers then give equal amount to trade with target and stoploss
#   for each trader.
#################################################################################
"""

import configparser, pickle
import time, os, sys
import pandas as pd
import math
import logging
import talib
import numpy as np
from datetime import datetime, date
# from kiteconnect import KiteConnect, KiteTicker

# Config file
# config = configparser.ConfigParser()
# config.read('/home/ubuntu/zerodha/config.cfg')

# # Initialize from config file
# API_KEY = config['KITE']['API_KEY']
# API_SECRET = config['KITE']['API_SECRET']
# ACCESS_TOKEN = config['LOGIN']['ACCESS_TOKEN']
# PUBLIC_TOKEN = config['LOGIN']['PUBLIC_TOKEN']
# BALANCE = config['PROFILE']['BALANCE']

BALANCE = 2000
init_bal = BALANCE
target_percent = 10
stoploss_percent = 7.5
stop_trading = False

# Set logging level
logging.basicConfig(level=logging.DEBUG)

# # Init KiteConnect
# kite = KiteConnect(api_key=API_KEY)

# # Set Access Token
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

# list to store current tradable stocks
selected_stocks = []

# dict to store all the dataframe
stock_df = {}
stock_list_name = []

# Trader class to handle all trading funcs
class RayTrader:

    def __init__(self, stock):
        self.stock_name = stock
        self.tradable = False
        self.open_trade = False
        self.leverage = 15
        self.target = 0.0
        self.amount_used = 0.0
        self.loss = 0.0
        self.source_prices = []
        self.renko_prices = []
        self.renko_directions = []
        self.oc_2 = []
        self.orders = []
        self.open = 0.0
        self.change = 0.0
        self.close = []
        self.long = False
        self.short = False
        self.pos = 0.0
        self.trail_pos = 0.0
        self.pos_qty = 1
        self.pnl = 0.0
        self.last_tick = {}
        self.buy_sell = 'N'
        self.unrealized_pnl = 0.0
        self.first_tick = True

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
        global BALANCE
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
                self.pnl = 0.0
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
                self.pnl = 0.0
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
        BALANCE += (self.pnl + self.amount_used)
        self.amount_used = 0.0
        # print("BALANCE: ", round(BALANCE,2))

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

    def get_tradable_stocks(self):
        count = 0
        for s in selected_stocks:
            stock_symbol = stock_list[token_list.index(s[0])]
            trader = class_object_dict[stock_symbol]
            if trader.tradable:
                count += 1
        return count

    def get_qty(self, price):
        global BALANCE
        # Depending on the number of stocks equally divide the amount
        # amount = float(BALANCE) / float(self.get_tradable_stocks())
        amount = BALANCE
        margin_amount = (amount) * self.leverage
        self.pos_qty = math.floor(margin_amount / price)
        if self.pos_qty>0:
            print("BALANCE: ", round(BALANCE, 2))
            self.amount_used = amount
            BALANCE -= amount
        return self.pos_qty

    def exit_trades(self, tick, timestamp):
        if tick == []:
            tick = self.last_tick
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

    def get_close_price(self):
        self.close.clear()

        for i in range(len(self.renko_prices)):
            self.close.append(float(self.renko_prices[i]))

        return np.array(self.close)

    def get_hma(self, price, timeperiod=14):
        # HMA= WMA(2*WMA(n/2) âˆ’ WMA(n)),sqrt(n))
        return (talib.WMA(
            2 * talib.WMA(price, timeperiod=math.floor(timeperiod / 2)) - talib.WMA(price, timeperiod=timeperiod),
            timeperiod=math.sqrt(timeperiod)))

    def get_rsi(self, price, timperiod=14):
        # return rsi
        return (talib.RSI(price, timperiod))

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
    #                                     trigger_price=(self.get_brick_size() * 2))

    #         logging.info("Order placed. ID is: {}".format(order_id))

    #         return order_id
    #     except Exception as e:
    #         logging.info("Order placement failed: {}".format(e.message))

    def get_oc2_price(self):
        # get the mean of open close
        self.oc_2.clear()

        for i in range(1, len(self.renko_prices)):
            j = i - 1
            self.oc_2.append((float(self.renko_prices[j]) + float(self.renko_prices[i])) / 2.0)

        return np.array(self.oc_2)

    def get_sma(self):
        # Calculate sma
        return talib.SMA(self.get_oc2_price(), timeperiod=8)

    def print_w(self, statement):
        order_dir_path = dir_path + self.stock_name + ".txt"
        with open(order_dir_path, "a") as text_file:
            print(statement, file=text_file)

    def on_tick(self, tick, timestamp):
        global BALANCE

        self.last_tick = tick
        # Once the tick is received now process it to decide to Buy or Sell
        # check time -->  if ear;ier than 9:15 store as open price
        # self.print_w("Stock : {0} Qty: {1}".format(self.stock_name,self.get_qty(float(tick['last_price']))))
        if self.first_tick:
            self.do_next(pd.Series([tick['last_price']]))
            self.open = float(tick['last_price'])
            self.cal_change(float(tick['last_price']))
            self.set_target_loss(float(tick['last_price']))
            self.first_tick = False
            self.print_w("Saving first tick as open: {0}".format(self.open))

        else:
            new_renko_bars, dir = self.do_next(pd.Series([tick['last_price']]))

            # Update BO also check if order got executed --> then dont do anything in this tick
            if self.open_trade:
                exe = self.update_bo(timestamp, float(tick['last_price']))
                if exe == 1:
                    return

            # If only new renko bars are formed
            if new_renko_bars != 0:
                self.unrealized_pnl = self.get_profit_loss(float(tick['last_price']))
                if self.unrealized_pnl is None:
                    self.unrealized_pnl = 0.0

                # Calculate % change
                self.cal_change(float(tick['last_price']))
                self.print_w("TimeStamp: {0} Number of Bars: {1} Unrealised PnL: {2} Change: {3}".format(timestamp,
                                                                                                         new_renko_bars,
                                                                                                         self.unrealized_pnl,
                                                                                                         self.change))

                # Update BO also check if order got executed --> then dont do anything in this tick
                # exe = self.update_bo(timestamp, float(tick['last_price']), new_renko_bars, int(dir))
                # if exe == 1:
                #     return

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
                                self.exit_trades(tick,timestamp)
                                # self.print_w("TimeStamp: {0} \tExit Short LTP: {1} RSI: {2} HMA: {3}".format(timestamp,
                                #                                                 tick['last_price'],round(rsi,2),round(hma,2)))
                                # self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, tick['last_price'],
                                #                                                  self.get_profit_loss(float(tick['last_price']))))
                                # self.pnl += self.get_profit_loss(float(tick['last_price']))
                                # self.short = False
                                # self.pos = 0.0
                                # self.unrealized_pnl = 0.0
                                # self.check_tradable()
                                # exit Short pos
                            if not stop_trading and self.buy_sell=='B':
                                qty = self.get_qty(float(tick['last_price']))
                                if qty > 0:
                                    self.pos = self.trail_pos = float(tick['last_price'])
                                    self.long = True
                                    self.place_bo(qty, float(tick['last_price']))
                                    self.print_w(
                                        "TimeStamp: {0} \tBuy Stock\t Qty: {1} LTP: {2} RSI: {3} HMA: {4}".format(timestamp,self.pos_qty,
                                                                                        tick['last_price'],round(rsi, 2),round(hma,2)))
                                    print("TimeStamp: {0} \tBuy Stock:{1}\t Qty: {2} LTP: {3}".format(timestamp, self.stock_name,
                                                                                                         self.pos_qty,tick['last_price']))

                    else:
                        last_renko_price = float(self.renko_prices[-1])
                        # check if sma has crossed the renko bar
                        if last_renko_price <= hma and rsi < 49.5 and self.short == False:
                            # check if stock already long
                            if self.long:
                                self.exit_trades(tick, timestamp)
                                # self.print_w("TimeStamp: {0} \tExit Long\t LTP: {1} RSI: {2} HMA: {3}".format(timestamp,
                                #                                                 tick['last_price'],round(rsi,2),round(hma,2)))
                                # self.print_w("From: {0} To: {1} Pnl: {2}".format(self.pos, tick['last_price'],
                                #                                                  self.get_profit_loss(float(tick['last_price']))))
                                # self.pnl += self.get_profit_loss(float(tick['last_price']))
                                # self.long = False
                                # self.pos = 0.0
                                # self.unrealized_pnl = 0.0
                                # self.check_tradable()
                                # exit Long pos
                            if not stop_trading and self.buy_sell=='S':
                                qty = self.get_qty(float(tick['last_price']))
                                if qty>0:
                                    self.pos = self.trail_pos = float(tick['last_price'])
                                    self.short = True
                                    self.place_bo(qty, float(tick['last_price']))
                                    self.print_w(
                                        "TimeStamp: {0} \tSell Stock!\t Qty: {1} LTP: {2} RSI: {3} HMA: {4}".format(timestamp,
                                                                            self.pos_qty,tick['last_price'],round(rsi,2),round(hma,2)))
                                    print("TimeStamp: {0} \tSell Stock:{1}\t Qty: {2} LTP: {3}".format(timestamp, self.stock_name,
                                                                                                         self.pos_qty,tick['last_price']))


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


# Func to select stocks and mark tradable
def select_stocks(ticks):
    global selected_stocks
    global stock_df, stock_list_name
    selected_stocks = []
    # check which stocks follow these the day before
    # Open = Low or Open = High
    for df_stock in stock_list_name:
        df = stock_df[df_stock]
        df = df[-375-15:-15]
        df_high = df['High'].max()
        df_low = df['Low'].min()
        df_open = df['Open'].iloc[0]

        if df_open == df_high:
            tup = (df_stock, 'S')
            selected_stocks.append(tup)
        elif df_open == df_low:
            tup = (df_stock, 'B')
            selected_stocks.append(tup)

        # print("Stock: ",df_stock)
        # print(df.head())
        # print(df.tail())
        # print("High:",df_high,"Low",df_low,"Open",df_open)
    # print(day_index)
    print("Selected Stocks: ", selected_stocks)

    for s in selected_stocks:
        trader_obj = class_object_dict[s[0]]
        trader_obj.tradable = True
        trader_obj.buy_sell = s[1]


# Func to select stocks and mark tradable
# Based on the direction of nifty
def select_stocks_nifty(ticks, return_no=3):
    global selected_stocks
    # list to store (instrument token, change) tuple
    stock_change = []
    nifty = 0.0
    # Now seperate the ticks
    for tick in ticks:
        # Get the nifty-50 % change
        if int(tick['instrument_token']) == 256265:
            nifty = float(tick['change'])
            print("nifty:", nifty)
        else:
            # Store the rest of the change and sort it out
            tup = (int(tick['instrument_token']), round(float(tick['change']),2))
            stock_change.append(tup)

    # Now check dir of nifty
    if nifty > 0.25:
        # Strong Uptrend
        print("Uptrend")
        # if +ve then sort stock with reverse=True and use change as key
        stock_change.sort(key=sortSecond, reverse=True)
        # Now mark those stocks tradable
        selected_stocks = stock_change[0:return_no]

    elif nifty < -0.25:
        # Strong Downtrend
        print("Downtrend")
        # if -ve then sort stock and use change as key
        stock_change.sort(key=sortSecond)
        # Now mark those stocks tradable
        selected_stocks = stock_change[0:return_no]

    else:
        # Dont know the trend
        stock_change.sort(key=sortSecond, reverse=True)
        # Now mark First 2 and last 2 stocks tradable
        selected_stocks = stock_change[0:2]

        stock_change.sort(key=sortSecond)
        selected_stocks.extend(stock_change[0:2])

        print("Selected Stocks: ", selected_stocks)

    for stock in selected_stocks:
        stock_token = stock[0]
        stock_symbol = stock_list[token_list.index(stock_token)]
        trader_obj = class_object_dict[stock_symbol]
        trader_obj.tradable = True


# -------------------- Create and load Instances ------------------

historical_dir = ".\hist_1_data\\"
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
    index = pd.to_datetime("13-03-2019 09:29:00 AM", format='%d-%m-%Y %I:%M:%S %p')
    df = df[:index]
    # print(df.tail())
    return df


# Create class objects of all the stocks that are subscribed to
for f in stock_list:
    trader = RayTrader(f)
    trader.set_brick_size(auto=False, brick_size=round(opt_renko_box[f], 4))
    hist_path = historical_dir + f + file_end
    df = csv_to_df(hist_path)
    stock_df[f] = df
    stock_list_name.append(f)
    # trader.set_brick_size(auto = True, HLC_history = df[["High", "Low", "Close"]])
    trader.build_history(prices=df.Close)
    trader.print_w("Stock: {0}".format(os.path.split(f)[-1]))
    trader.print_w("Opt Renko size: {0}".format(trader.get_brick_size()))
    class_object_dict[f] = trader

# ------------------ Market Watch ----------------------------

# variable to run startup code
startup_code = True
close_code = True

# Initialise KiteTicker -> Live Ticks
# kws = KiteTicker(API_KEY, ACCESS_TOKEN)


def on_ticks(ws, ticks, timestamp):
    global startup_code, close_code, stop_trading
    # Callback to receive ticks.
    # timestamp = datetime.now()
    # First time send the ticks to select the stocks that will be marked tradable
    if startup_code:
        print("Running Startup Code")
        select_stocks(ticks)
        startup_code = False

    else:
        # Zerodha squareoff time is 3:20
        # So dont take any trades after 3 pm
        if timestamp.hour < 11:
            # Now seperate the ticks and call on_tick of repective objects
            for tick in ticks:
                if int(tick['instrument_token']) != 256265:
                    stock_symbol = stock_list[token_list.index(tick['instrument_token'])]
                    trader_obj = class_object_dict[stock_symbol]
                    if trader_obj.tradable:
                        if stop_trading:
                            trader_obj.exit_trades(tick, timestamp)
                        trader_obj.on_tick(tick, timestamp)
                # stock_file_path = dir_path + "/" + stock_symbol + ".csv"
                # df = pd.DataFrame(tick, index=[timestamp])
                # df.to_csv(stock_file_path, mode='a', header=False)

        # Stop the websocket after 3:30 pm
        if timestamp.hour == 11:
            if close_code:
                # print the final PnL
                for trader in class_object_dict.values():
                    if trader.open_trade:
                        trader.exit_trades(tick=[],timestamp=timestamp)
                    if trader.unrealized_pnl != 0.0:
                        trader.pnl += trader.unrealized_pnl
                        trader.unrealized_pnl = 0.0
                    trader.print_w("Final PnL: {0}".format(trader.pnl))
                # exit the program
                close_code = False
                stop_trading = True


# def on_connect(ws, response):
#     # Callback on successful connect.
#     # Subscribe to a list of instrument_tokens {All stocks present in historical data}
#     sub_list = token_list
#     # add nifty-50 to the list
#     sub_list.append(256265)
#     ws.subscribe(sub_list)

    # Set Mode QUOTE
    # ws.set_mode(ws.MODE_LTP, token_list)


# def on_close(ws, code, reason):
#     # On connection close stop the main loop
#     # Reconnection will not happen after executing `ws.stop()`
#     ws.stop()

# Assign the callbacks.
# kws.on_ticks = on_ticks
# kws.on_connect = on_connect
# kws.on_close = on_close

# Infinite loop on the main thread. Nothing after this will run.
# You have to use the pre-defined callbacks to manage subscriptions.
# kws.connect()
print("Loading tick data")
data_dir = [ ".\\live_ticks\\2019-03-13\\" + str(i) + ".out" for i in range(17,377)]

print("Initial Balance: ",BALANCE)

for in_file in data_dir:
    print("Processing :", in_file)
    # unpickle each file and the feed to on_ticks()
    live_ticks = []
    # Read from the file the stock list
    with open(in_file, 'rb') as fp:
        live_ticks = pickle.load(fp)

    for ticks in live_ticks:
        timestamp = datetime.fromtimestamp(os.path.getmtime(in_file))
        on_ticks(0, ticks, timestamp)
        if stop_trading:
            break

    if stop_trading:
        break

print("Final Balance: ",round(BALANCE,2))
print("Overall Pnl: ",round(BALANCE-init_bal,2))