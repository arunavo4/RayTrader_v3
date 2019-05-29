import trader_env
import trader_data
import random as rand
from statistics import mean
import numpy as np
import neat
import pickle

file_name = "G:\\AI Trading\\Code\\RayTrader_v3\\HistoricalData\\Min_data\\ADANIPORTS-EQ.csv"
data = trader_data.csv_to_df(file_name)
train_data, test_data = trader_data.split_data(data)

env = trader_env.Weighted_Unrealized_BS_Env(train_data)

max_env_steps = len(env.data) - 1


k=0

for i in range(len(env.data)):
    print(env.t,env.data.index[env.t])
    act = rand.randint(0,2)
    print("act: ",act)
    for s in env.signals.values():
        print(s[env.t:env.t+1])
        break
    obs, reward, done, info = env.step(act)
    print(train_data[env.t:env.t+1])
    print(obs, reward, done, info)
    print(env.action_record)
    print("")
    if (env.t)%375 == 0:
        k += 1

    if k == 2:
        print(env.daily_profit_per)
        print(round(mean(env.daily_profit_per),3))
        break



#
# size_data = len(train_data) - 1
# print(size_data)
#
# print(len(train_data[40:]))
#
#
# print(train_data[40:41])
# for s in env.signals.values():
#     print(s[40:41])
    # break
# inputs = trader_data.get_inputs(signals,-1)
# print(inputs)
