import pandas as pd
import glob
import os

# Func to load the csv and return a dataframe
def csv_to_df(csv_file):
    df = pd.read_csv(csv_file,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'])

    # Drop the header
    df = df.drop(df.index[0])
    # Drop Duplicates
    df = df.drop_duplicates('Date')
    # parse date
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %I:%M:%S %p')
    # Set index
    df = df.set_index('Date')
    # sort the index according to date
    df = df.sort_index()
    # reindex the df to make it look right
    df = df.reindex(index=df.index[::-1])
    # Change Format
    df.index = df.index.strftime('%d-%m-%Y %I:%M:%S %p')
    df.index.name = 'Date'
    # print(df.head())
    # print(df.tail())
    df.to_csv(csv_file, header=True)



historical_dir = ".\Min_data\*.csv"
data_dir = glob.glob(historical_dir)

for f in data_dir:
    print("Fixing :", os.path.split(f)[-1])
    csv_to_df(f)
