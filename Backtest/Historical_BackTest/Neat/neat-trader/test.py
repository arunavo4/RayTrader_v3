import trader_env
import trader_data
import numpy as np
import neat
import pickle

file_name = "G:\\AI Trading\\Code\\RayTrader_v3\\HistoricalData\\Min_data\\ADANIPORTS-EQ.csv"
data = trader_data.csv_to_df(file_name)
train_data, test_data = trader_data.split_data(data)

print(train_data[40:41])
signals = trader_data.get_signals(data)
for s in signals.values():
    print(s[40:41])
    # break
# inputs = trader_data.get_inputs(signals,-1)
# print(inputs)
