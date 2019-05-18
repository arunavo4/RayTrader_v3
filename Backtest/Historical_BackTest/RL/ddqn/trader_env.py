from statistics import mean
from collections import deque
import numpy as np
import pandas as pd

# logging
# importing module
import logging

formatter = logging.Formatter('%(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


logger = setup_logger('env_logger', 'env.log')
logger.info('Env Logger!')

# Class Definition for Trader


"""
####Buy/Sell (Short Sell) Env (Realized profits reward)
Using profit % as reward and not profits. Also has the option to short sell.
No intermediate rewards given.
"""


class Realized_BS_Env:

    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = int(0)
        self.done = False
        self.short = False
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.history = [int(0) for _ in range(self.history_t)]
        return [self.position_value] + self.history  # obs

    def step(self, act):
        reward = 0
        profit_percent = float(0)
        self.position_record = ""
        self.action_record = ""

        if (self.t + 1) % 375 == 0:
            # auto square-off
            # Check of trades taken
            if len(self.positions) != 0:
                logger.info("Auto Squareoff")
                if self.short:
                    act = 1
                else:
                    act = 2

        # act = 0: hold, 1: buy, 2: sell
        if act == 1:  # buy
            if len(self.positions) == 0:
                # Going Long
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                reward = 1
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Long",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            elif not self.short and len(self.positions) != 0:
                # If stock has been already long
                reward = -1
                message = "Dont try to go long more than once!"
                self.action_record = message
                logger.info(message)

            else:
                # exit from Short Sell
                profits = 0
                for p in self.positions:
                    profits += (p - float(self.data.iloc[self.t, :]['Close']))

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.profit_per += round(profit_percent, 3)
                reward += profit_percent * 10
                # Save the record of exit
                self.position_record = "Timestep {}:==: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self.data.index[self.t],
                    len(self.positions) * -1,
                    round(mean(self.positions), 2),
                    round(float(self.data.iloc[self.t, :]['Close']), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        elif act == 0:  # hold
            # No rewards
            if len(self.positions) == 0:
                self.action_record = "Thinking for next move!"
            else:
                self.action_record = "Holding!"

        elif act == 2:  # sell
            if len(self.positions) == 0:
                # Going Short
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                reward = 1
                self.short = True
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            elif self.short and len(self.positions) != 0:
                # If stock has been already short
                reward = -1
                message = "Dont try to short more than once!"
                self.action_record = message
                logger.info(message)
            else:
                # exit from the Long position
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.profit_per += round(profit_percent, 3)
                reward += profit_percent * 10
                # Save the record of exit
                self.position_record = "Timestep {}:==: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self.data.index[self.t],
                    len(self.positions),
                    round(mean(self.positions), 2),
                    round(float(self.data.iloc[self.t, :]['Close']), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit Long",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        # set next time
        self.t += 1
        if self.t % 375 == 0:
            self.daily_profit_per.append(round(self.profit_per, 3))
            self.profit_per = 0.0

        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clip reward
        reward = round(reward, 3)

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done


"""
####Buy/Sell (Short Sell) Env (Unrealized profits reward)
Using profit % as reward and not profits. Also has the option to short sell.
"""


class Unrealized_BS_Env:

    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = int(0)
        self.done = False
        self.short = False
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.history = [int(0) for _ in range(self.history_t)]
        return [self.position_value] + self.history  # obs

    def step(self, act):
        reward = 0
        profit_percent = float(0)
        self.position_record = ""
        self.action_record = ""

        if (self.t + 1) % 375 == 0:
            # auto squareoff
            # Check of trades taken
            if len(self.positions) != 0:
                logger.info("Auto Squareoff")
                if self.short:
                    act = 1
                else:
                    act = 2

        # act = 0: hold, 1: buy, 2: sell
        if act == 1:  # buy
            if len(self.positions) == 0:
                # Going Long
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                reward = 1
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Long",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            elif not self.short and len(self.positions) != 0:
                # If stock has been already long
                reward = -1
                message = "Dont try to go long more than once!"
                self.action_record = message
                logger.info(message)

            else:
                # exit from Short Sell
                profits = 0
                for p in self.positions:
                    profits += (p - float(self.data.iloc[self.t, :]['Close']))

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.profit_per += round(profit_percent, 3)
                reward += profit_percent * 10
                # Save the record of exit
                self.position_record = "Timestep {}:==: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self.data.index[self.t],
                    len(self.positions) * -1,
                    round(mean(self.positions), 2),
                    round(float(self.data.iloc[self.t, :]['Close']), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        elif act == 0:  # hold
            if len(self.positions) > 0:
                profits = 0
                for p in self.positions:
                    if self.short:
                        profits += (p - float(self.data.iloc[self.t, :]['Close']))
                    else:
                        profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += profit_percent
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Hold",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            else:
                self.action_record = "Thinking for next move!"

        elif act == 2:  # sell
            if len(self.positions) == 0:
                # Going Short
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                reward = 1
                self.short = True
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            elif self.short and len(self.positions) != 0:
                # If stock has been already short
                reward = -1
                message = "Dont try to short more than once!"
                self.action_record = message
                logger.info(message)

            else:
                # exit from the Long position
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.profit_per += round(profit_percent, 3)
                reward += profit_percent * 10
                # Save the record of exit
                self.position_record = "Timestep {}:==: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self.data.index[self.t],
                    len(self.positions),
                    round(mean(self.positions), 2),
                    round(float(self.data.iloc[self.t, :]['Close']), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit Long",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        # set next time
        self.t += 1
        if self.t % 375 == 0:
            self.daily_profit_per.append(round(self.profit_per, 3))
            self.profit_per = 0.0

        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clip reward
        reward = round(reward, 3)

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done


"""
####Buy/Sell (Short Sell) Env (Unrealized *weighted* profits reward)
Using profit % as reward (*weighted with time decay*) and not profits
Also has the option to short sell.
"""


class Weighted_Unrealized_BS_Env:

    def __init__(self, data, history_t=90):
        self.decay_rate = 1e-2
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = int(0)
        self.done = False
        self.short = False
        self.amt = 2000.0
        self.profit_per = float(0.0)
        self.daily_profit_per = []
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)
        self.history = [int(0) for _ in range(self.history_t)]
        return [self.position_value]  # obs

    def getReward(self, reward):
        stale_reward = self.rewards.popleft()
        self.sum = self.sum - np.exp(-1 * self.decay_rate) * stale_reward
        self.sum = self.sum * np.exp(-1 * self.decay_rate)
        self.sum = self.sum + reward
        self.rewards.append(reward)
        return self.sum / self.denominator

    def step(self, act):
        reward = 0
        profit_percent = float(0)
        self.position_record = ""
        self.action_record = ""

        if (self.t + 1) % 375 == 0:
            # auto squareoff
            # Check of trades taken
            if len(self.positions) != 0:
                logger.info("Auto Squareoff")
                if self.short:
                    act = 1
                else:
                    act = 2

        # act = 0: hold, 1: buy, 2: sell
        if act == 1:  # buy
            if len(self.positions) == 0:
                # Going Long
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                reward = 1
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Long",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            elif not self.short and len(self.positions) != 0:
                # If stock has been already long
                reward = 0
                message = "Don't try to go long more than once!"
                self.action_record = message
                logger.info(message)

            else:
                # exit from Short Sell
                profits = 0
                for p in self.positions:
                    profits += (p - float(self.data.iloc[self.t, :]['Close']))

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.amt += self.amt*(profit_percent/100)
                self.profit_per += round(profit_percent, 3)
                reward += self.getReward(profit_percent) * 100
                # Save the record of exit
                self.position_record = "Timestep {}:==: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self.data.index[self.t],
                    len(self.positions) * -1,
                    round(mean(self.positions), 2),
                    round(float(self.data.iloc[self.t, :]['Close']), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        elif act == 0:  # hold
            if len(self.positions) > 0:
                profits = 0
                for p in self.positions:
                    if self.short:
                        profits += (p - float(self.data.iloc[self.t, :]['Close']))
                    else:
                        profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += self.getReward(profit_percent)
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Hold",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            else:
                self.action_record = "Thinking for next move!"

        elif act == 2:  # sell
            if len(self.positions) == 0:
                # Going Short
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                reward = 1
                self.short = True
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

            elif self.short and len(self.positions) != 0:
                # If stock has been already short
                reward = 0
                message = "Dont try to short more than once!"
                self.action_record = message
                logger.info(message)

            else:
                # exit from the Long position
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                self.amt += self.amt*(profit_percent/100)
                self.profit_per += round(profit_percent, 3)
                reward += self.getReward(profit_percent) * 100
                # Save the record of exit
                self.position_record = "Timestep {}:==: Qty : {} ; Avg: {} ; Ltp: {} ; P&L: {} ; %Chg: {}".format(
                    self.data.index[self.t],
                    len(self.positions),
                    round(mean(self.positions), 2),
                    round(float(self.data.iloc[self.t, :]['Close']), 2),
                    round(profits, 2),
                    round(profit_percent, 3))
                self.profits += profits
                self.positions = []
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit Long",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        # set next time
        self.t += 1
        if self.t % 375 == 0:
            reward += self.profit_per * 1000        #Bonus for making a profit at the end of the day
            self.daily_profit_per.append(round(self.profit_per, 3))
            self.profit_per = 0.0

        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clip reward
        reward = round(reward, 3)

        return [self.position_value], reward, self.done  # obs, reward, done
