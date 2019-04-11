"""
# Trying to fix the lag in the prediction
"""

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
    # df = pd.read_csv(csv_file,
    #                    names=['Date', 'Open', 'High', 'Low', 'Close', 'Vol'])
    df = pd.read_csv(csv_file,
                     names=['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Vol'])

    # Drop the header
    df = df.drop(df.index[0])
    # parse date
    #drop nan
    df = df.dropna()
    # df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %I:%M:%S %p')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
    # Set index
    df = df.set_index('Date')
    # reindex the df to make it look right
    df = df.reindex(index=df.index[::-1])
    # Drop the Vol
    df = df[['Open', 'High', 'Low', 'Close']]
    return df

# import datasetTrader/stock_prediction/data/data_day/ADANIPORTS.NS.csv
file_name = 'G:\AI Trading\Code\RayTrader_v3\Trader\stock_prediction\data\data_day\ADANIPORTS.NS.csv'
df_stock = csv_to_df(file_name)

# data scaling (normalizing)
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return df

df_stock_norm = df_stock.copy()
df_stock_norm = normalize_data(df_stock_norm)

# Splitting the dataset into Train, Valid & test data
valid_set_size_percentage = 10
test_set_size_percentage = 10
seq_len = 20  # taken sequence length as 20

def addNameToTensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)

def load_seq_data(stock, seq_len):
    # data_raw = stock.as_matrix()
    data_raw = stock.values
    data = []
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])
    data = np.array(data)
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0])) - 8
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size + 22)
    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]
    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]
    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

def load_data(stock, seq_len):
    # data_raw = stock.as_matrix()
    data_raw = stock.values
    data = data_raw
    # for index in range(len(data_raw) - seq_len):
    #     data.append(data_raw[index: index + seq_len])
    # data = np.array(data_raw)
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)
    x_train = data[:train_set_size, :]
    y_train = data[:train_set_size, 3]
    x_valid = data[train_set_size:train_set_size + valid_set_size, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, 3]
    x_test = data[train_set_size + valid_set_size:, :]
    y_test = data[train_set_size + valid_set_size:, 3]
    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

x_train, y_train, x_valid, y_valid, x_test, y_test = load_seq_data(df_stock_norm, seq_len)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid.shape = ', x_valid.shape)
print('y_valid.shape = ', y_valid.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

"""Building the Model"""

# parameters & Placeholders
n_steps = seq_len - 1
n_inputs = 4
n_neurons = 200
n_outputs = 4
n_layers = 2
learning_rate = 0.001
batch_size = 50
n_epochs = 100
train_set_size = x_train.shape[0]
test_set_size = x_test.shape[0]


""" Keras Model """

np.random.seed(1234)

input_tensor = tf.keras.Input((19, 4,))  #open high and volume as input
x = input_tensor
rnn_size = n_neurons
W_INIT = 'he_normal'
for i in range(3):
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32*2**i,return_sequences=True,
                          go_backwards=True, kernel_initializer=W_INIT))(x)
x=tf.keras.layers.Dropout(0.25)(x)
x=tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(4, activation='linear',kernel_initializer='normal')(x)
model = tf.keras.models.Model(inputs=input_tensor, outputs=x)
adam= tf.keras.optimizers.Adam(lr=0.0001)
model.compile(loss='mse',
              optimizer=adam)


checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('stateless_BiGRU_v2_pretrained.hdf5'
                                      , monitor='val_loss', verbose=0, save_best_only=True,
                                      save_weights_only=False, mode='auto', period=1)

# model = tf.keras.models.load_model('stateful_BiGRU_pretrained.hdf5')
# epochs = 100
# for i in range(epochs):
#     model.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[checkpoint_callback])
#     model.reset_states()
#
# model.fit(
#     x_train, y_train,
#     batch_size=batch_size, epochs=n_epochs,verbose=1,
#     validation_data=(x_valid,y_valid),initial_epoch = 0,
#     shuffle=True, callbacks=[checkpoint_callback]
# )

model = tf.keras.models.load_model('stateful_SimpleRNN_pretrained.hdf5')

y_test_pred = model.predict(x_test)

print(y_test_pred.shape)

# ploting the graph
comp = pd.DataFrame({'Column1': y_test[:,3], 'Column2': y_test_pred[:,3]})
plt.figure(figsize=(10, 5))
plt.plot(comp['Column1'], color='blue', label='Target')
plt.plot(comp['Column2'], color='black', label='Prediction')
plt.legend()
plt.show()