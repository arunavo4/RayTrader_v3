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
signals = trader_data.get_signals(data)

with open('signals.pkl', 'wb') as output:
    pickle.dump(signals, output, 1)
