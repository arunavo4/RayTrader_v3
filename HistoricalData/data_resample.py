
import pandas as pd
import glob
import os

# Func to load the csv and return a dataframe
def csv_to_df(csv_file):
    df = pd.read_csv(csv_file,
                       names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'])

    #Drop the header
    df = df.drop(df.index[0])
    #parse date
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %I:%M:%S %p')
    # Set index
    df = df.set_index('Date')
    #reindex the df to make it look right
    try:
        df = df.reindex(index=df.index[::-1])
    except:
        dup = df.index.duplicated()
        print(dup.index(True))

    #Drop the Vol
    # df = df[['Open', 'High', 'Low', 'Close']]
    return df


def data_resample(file_name):
    file_new = './Min_10_data/' + os.path.split(file_name)[-1]

    data = csv_to_df(file_name)
    # data = data.convert_objects(convert_numeric=True)
    data['Open'] = pd.to_numeric(data['Open'])
    data['High'] = pd.to_numeric(data['High'])
    data['Low'] = pd.to_numeric(data['Low'])
    data['Close'] = pd.to_numeric(data['Close'])
    data['Vol'] = pd.to_numeric(data['Vol'])
    # print(data.dtypes)
    # print("1Min data",data.head())

    ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Vol': 'sum'}

    # adding the base 15 adds an offset to the time
    data = data.resample('10Min', base=15).apply(ohlc_dict).dropna(how='any')

    cols = ['Open', 'High', 'Low', 'Close', 'Vol']
    data = data[cols]
    # reindex it to look it like others
    data = data.reindex(index=data.index[::-1])
    # print("15 Min data",data.head())
    data.index = data.index.strftime('%d-%m-%Y %I:%M:%S %p')
    data.index.name = 'Date'
    data.to_csv(file_new, header=True)


#--------------------- Main --------------------------

old = glob.glob(".\Min_data\*.csv")

for f in old:
    print("Resampling :",os.path.split(f)[-1])
    data_resample(f)
    # break

