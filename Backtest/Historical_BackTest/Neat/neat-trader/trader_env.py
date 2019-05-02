from statistics import mean
from collections import deque
import numpy as np
import pandas as pd

# logging
# importing module
import logging

# Create and configure logger
logging.basicConfig(filename="stock.log",
                    format='%(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

# Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

# Test messages
logger.info("Logging!")


# Class Definition for Trader
"""
# Basic Environment
"""


class Environment:

    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = int(0)
        self.done = False
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.history = [int(0) for _ in range(self.history_t)]
        return [self.position_value] + self.history  # obs

    def step(self, act):
        reward = 0

        # act = 0: hold, 1: buy, 2: exit
        if act == 1:  # buy
            self.positions.append(float(self.data.iloc[self.t, :]['Close']))
        elif act == 2:  # exit
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                reward += profits
                self.profits += profits
                self.positions = []

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clipping reward
        if reward > 0:
            reward = 1
        elif reward < 0:
            reward = -1

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done


"""
####Buy/Exit Env (Realized profits reward)
Using profit % as reward and not profits. No intermediate rewards given.
"""


class Realized_BE_Env:

    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = int(0)
        self.done = False
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

        # act = 0: hold, 1: buy, 2: exit
        if act == 1:  # buy
            self.positions.append(float(self.data.iloc[self.t, :]['Close']))
            message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Buy", round(reward, 3))
            self.action_record = message
            logger.info(message)

        elif act == 0:  # hold
            # no rewards
            if len(self.positions) == 0:
                self.action_record = "Thinking for next move!"
            else:
                self.action_record = "Holding!"

        elif act == 2:  # exit
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += profit_percent
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
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clip reward
        reward = round(reward, 3)

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done


"""
####Buy/Exit Env (Unrealized profits reward)
Using profit % as reward and not profits
"""


class Unrealized_BE_Env:

    def __init__(self, data, history_t=90):
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = int(0)
        self.done = False
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

        # act = 0: hold, 1: buy, 2: exit
        if act == 1:  # buy
            self.positions.append(float(self.data.iloc[self.t, :]['Close']))
            message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Buy", round(reward, 3))
            self.action_record = message
            logger.info(message)

        elif act == 0:  # hold
            if len(self.positions) > 0:
                profits = 0
                for p in self.positions:
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

        elif act == 2:  # exit
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += profit_percent
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
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clip reward
        reward = round(reward, 3)

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done


"""
####Buy/Exit Env (Unrealized *weighted* profits reward)
Using profit % as reward (*weighted with time decay*) and not profits
"""


class Weighted_Unrealized_BE_Env:

    def __init__(self, data, history_t=90):
        self.decay_rate = 1e-2
        self.data = data
        self.history_t = history_t
        self.reset()

    def reset(self):
        self.t = int(0)
        self.done = False
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)
        self.history = [int(0) for _ in range(self.history_t)]
        return [self.position_value] + self.history  # obs

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

        # act = 0: hold, 1: buy, 2: exit
        if act == 1:  # buy
            self.positions.append(float(self.data.iloc[self.t, :]['Close']))
            message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Buy", round(reward, 3))
            self.action_record = message
            logger.info(message)

        elif act == 0:  # hold
            if len(self.positions) > 0:
                profits = 0
                for p in self.positions:
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

        elif act == 2:  # exit
            if len(self.positions) == 0:
                reward = -1
            else:
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += self.getReward(profit_percent)
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
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Exit",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)

        # set next time
        self.t += 1
        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clip reward
        reward = round(reward, 3)

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done


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

        # act = 0: hold, 1: buy, 2: sell
        if act == 1:  # buy
            if self.short:
                # exit from Short Sell
                profits = 0
                for p in self.positions:
                    profits += (p - float(self.data.iloc[self.t, :]['Close']))

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += profit_percent
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

            else:
                # Going Long
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Long",
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
            if len(self.positions) == 0 or self.short:
                # Going Short
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                self.short = True
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)
            else:
                # exit from the Long position
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += profit_percent
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

        # act = 0: hold, 1: buy, 2: sell
        if act == 1:  # buy
            if len(self.positions) != 0 and self.short:
                # exit from Short Sell
                profits = 0
                for p in self.positions:
                    profits += (p - float(self.data.iloc[self.t, :]['Close']))

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += profit_percent
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

            else:
                # Going Long
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Long",
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
            if len(self.positions) == 0 or self.short:
                # Going Short
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                self.short = True
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)
            else:
                # exit from the Long position
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += profit_percent
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
        self.profits = int(0)
        self.positions = []
        self.position_value = int(0)
        self.action_record = ""
        self.position_record = ""
        self.rewards = deque(np.zeros(1, dtype=float))
        self.sum = 0.0
        self.denominator = np.exp(-1 * self.decay_rate)
        self.history = [int(0) for _ in range(self.history_t)]
        return [self.position_value] + self.history  # obs

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

        # act = 0: hold, 1: buy, 2: sell
        if act == 1:  # buy
            if len(self.positions) != 0 and self.short:
                # exit from Short Sell
                profits = 0
                for p in self.positions:
                    profits += (p - float(self.data.iloc[self.t, :]['Close']))

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += self.getReward(profit_percent)
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

            else:
                # Going Long
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                self.short = False
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Long",
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
            if len(self.positions) == 0 or self.short:
                # Going Short
                self.positions.append(float(self.data.iloc[self.t, :]['Close']))
                self.short = True
                message = "Timestep {}:==: Action: {} ; Reward: {}".format(self.data.index[self.t], "Short",
                                                                           round(reward, 3))
                self.action_record = message
                logger.info(message)
            else:
                # exit from the Long position
                profits = 0
                for p in self.positions:
                    profits += (float(self.data.iloc[self.t, :]['Close']) - p)

                avg_profit = profits / len(self.positions)
                profit_percent = (avg_profit / mean(self.positions)) * 100
                reward += self.getReward(profit_percent)
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
        self.position_value = 0
        for p in self.positions:
            self.position_value += (float(self.data.iloc[self.t, :]['Close']) - p)
        self.history.pop(0)
        self.history.append(float(self.data.iloc[self.t, :]['Close']) - float(self.data.iloc[(self.t - 1), :]['Close']))

        # clip reward
        reward = round(reward, 3)

        return [self.position_value] + self.history, reward, self.done  # obs, reward, done
