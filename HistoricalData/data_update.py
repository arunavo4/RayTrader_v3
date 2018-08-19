"""
    This program will join the existing csv files containing the 1 min data
    with the new files that have been saved from Zerodha pi software.
    To keep the historical data up to date.

    New_Min_data + Min_data will be merged.
"""
import glob
import os
import pandas as pd


def merge_csv(file_old, file_new, file):
    file_old = './Min_data/' + file_old
    file_new = './Min_data_new/' + file_new

    data_old = pd.read_csv(file_old, index_col='Date',
                       names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'])

    data_new = pd.read_csv(file_new, index_col='Date',
                           names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'])

    if data_old.index[1] == data_new.index[1]:
        print("Up To Date", file)
        return 0

    data_new = data_new.drop(data_new.index[0])
    data_old = data_old.drop(data_old.index[0])

    merged = pd.concat([data_new, data_old]).drop_duplicates()

    # Reverse the data to make it look right
    # merged = merged.reindex(index=merged.index[::-1])

    merged.to_csv(file_old, header=True)

    print("Merged! ", file)
    return 0

#----------------------------main--------------------------

old = glob.glob(".\Min_data\*.csv")
new = glob.glob(".\Min_data_new\*.csv")

data_dir_old = []
data_dir_new = []

for f in old:
    path = os.path.split(f)
    data_dir_old.append(path[-1])

for f in new:
    path = os.path.split(f)
    data_dir_new.append(path[-1])

for file_n in data_dir_new:
    for file_o in data_dir_old:
        if file_o == file_n:
            merge_csv(file_o, file_n, file_n)