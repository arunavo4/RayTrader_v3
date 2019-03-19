# import all libraries
import warnings
import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
tf.logging.info('TensorFlow')
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.info('TensorFlow')


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
    df = df.reindex(index=df.index[::-1])
    print(df.tail())
    #Drop the Vol
    df = df[['Open', 'High', 'Low', 'Close']]
    return df


# import datasetTrader/model/ADANIPORTS-EQ
file_name = '.\data\ADANIPORTS-EQ.csv'
df_stock = csv_to_df(file_name)
min_max_scaler = sklearn.preprocessing.MinMaxScaler()


# data rescaling back(de-norm)
def de_norm_data(df):
    global min_max_scaler
    df['Open'] = min_max_scaler.inverse_transform(df.Open.values.reshape(-1, 1))
    df['High'] = min_max_scaler.inverse_transform(df.High.values.reshape(-1, 1))
    df['Low'] = min_max_scaler.inverse_transform(df.Low.values.reshape(-1, 1))
    df['Close'] = min_max_scaler.inverse_transform(df['Close'].values.reshape(-1, 1))
    return df


# data scaling (normalizing)
def normalize_data(df):
    global min_max_scaler
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return df


df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

df_stock_denorm = df_stock_norm.copy()
df_stock_denorm = de_norm_data(df_stock_denorm)

# Splitting the dataset into Train, Valid & test data
valid_set_size_percentage = 10
test_set_size_percentage = 1
seq_len = 20  # taken sequence length as 20


def load_data(stock, seq_len):
    data_raw = stock.as_matrix()
    # data_raw = stock.values
    data = []
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])
    data = np.array(data)
    print("data raw shape",data_raw.shape)
    print("data shape:",data.shape)
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]
    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]


x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ', x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

tf.reset_default_graph()

# Fitting the model
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], 'G:\AI Trading\Code\RayTrader_v3\Trader\model\ADANIPORTS-EQ')
    graph = tf.get_default_graph()
    print(graph.get_operations())
    X_graph = graph.get_tensor_by_name('myInput:0')
    output_graph = graph.get_tensor_by_name('myOutput:0')
    # Predictions
    y_test_pred = sess.run(output_graph,
               feed_dict={X_graph: x_test})

# checking prediction output nos
print(y_test_pred.shape)


#original data
print("Original data")
print(df_stock.head())
print(df_stock.tail())

#denorm data
print("De-Normalized data")
print(df_stock_denorm.head())
print(df_stock_denorm.tail(10))

#normal data
print("Normalized data")
print(df_stock_norm.head())
print(df_stock_norm.tail())

print("\n\n")
# Scale back to actual values
# x_test = min_max_scaler.inverse_transform(x_test)
y_test = min_max_scaler.inverse_transform(y_test)
y_test_pred = min_max_scaler.inverse_transform(y_test_pred)

# print("Input data: ",x_test)

print("Actual: ", y_test[:,3])
print("Predicated: ",y_test_pred[:,3])
# plotting the graph
comp = pd.DataFrame({'Column1': y_test[:, 3], 'Column2': y_test_pred[:, 3]})
plt.figure(figsize=(10, 5))
plt.plot(comp['Column1'], color='blue', label='Target')
plt.plot(comp['Column2'], color='black', label='Prediction')
plt.legend()
plt.show()



