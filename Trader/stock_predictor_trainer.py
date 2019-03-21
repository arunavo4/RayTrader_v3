# import all libraries
import sklearn.preprocessing
import matplotlib.pyplot as plt

import time, os, sys
import warnings
import pandas as pd
import sklearn
import sklearn.preprocessing
import glob
import tensorflow as tf
import numpy as np
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
    #Drop the Vol
    df = df[['Open', 'High', 'Low', 'Close']]
    return df


# data scaling (normalizing)
def normalize_data(df):
    min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    df['Open'] = min_max_scaler.fit_transform(df.Open.values.reshape(-1, 1))
    df['High'] = min_max_scaler.fit_transform(df.High.values.reshape(-1, 1))
    df['Low'] = min_max_scaler.fit_transform(df.Low.values.reshape(-1, 1))
    df['Close'] = min_max_scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    return df

# function to get the next batch
index_in_epoch = 0

# Splitting the dataset into Train, Valid & test data
valid_set_size_percentage = 10
test_set_size_percentage = 10
seq_len = 20  # taken sequence length as 20


def addNameToTensor(someTensor, theName):
    return tf.identity(someTensor, name=theName)


def load_data(stock, seq_len):
    # data_raw = stock.as_matrix()
    data_raw = stock.values
    data = []
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])
    data = np.array(data)
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


def get_next_batch(batch_size, x_train, perm_array, y_train):
    global index_in_epoch
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > x_train.shape[0]:
        np.random.shuffle(perm_array)  # shuffle permutation array
        start = 0  # start next epoch
        index_in_epoch = batch_size
    end = index_in_epoch
    return x_train[perm_array[start:end]], y_train[perm_array[start:end]]


def main(file_name):
    global index_in_epoch
    # import dataset
    export_dir = 'model_30/' + file_name
    file_name = '.\\data_30\\' + file_name + '.csv'
    df_stock = csv_to_df(file_name)

    df_stock_norm = df_stock.copy()
    df_stock_norm = normalize_data(df_stock_norm)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df_stock_norm, seq_len)

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
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="myInput")
    y = tf.placeholder(tf.float32, [None, n_outputs], name="Y")

    # function to get the next batch
    index_in_epoch = 0
    perm_array = np.arange(x_train.shape[0])
    np.random.shuffle(perm_array)

    # RNN
    # layers = [tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
    #           for layer in range(n_layers)]
    # LSTM
    layers = [tf.contrib.rnn.BasicLSTMCell(num_units=n_neurons, activation=tf.nn.elu)
              for layer in range(n_layers)]

    # LSTM with peephole connections
    # layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons,
    #                                  activation=tf.nn.leaky_relu, use_peepholes = True)
    #          for layer in range(n_layers)]

    # GRU
    # layers = [tf.contrib.rnn.GRUCell(num_units=n_neurons, activation=tf.nn.leaky_relu)
    #          for layer in range(n_layers)]

    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    rnn_outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)
    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    outputs = outputs[:,n_steps-1,:] # keep only last output of sequence

    addNameToTensor(outputs,"myOutput")

    # Cost function
    loss = tf.reduce_mean(tf.square(outputs - y))

    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    # Fitting the model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for iteration in range(int(n_epochs * train_set_size / batch_size)):
            x_batch, y_batch = get_next_batch(batch_size,x_train,perm_array,y_train)  # fetch the next training batch
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch})
            if iteration % int(5 * train_set_size / batch_size) == 0:
                mse_train = loss.eval(feed_dict={X: x_train, y: y_train})
                mse_valid = loss.eval(feed_dict={X: x_valid, y: y_valid})
                print('%.2f epochs: MSE train/valid = %.6f/%.6f' % (
                    iteration * batch_size / train_set_size, mse_train, mse_valid))

        # Save the variables to disk.
        tf.saved_model.simple_save(sess,
                export_dir,
                inputs={"myInput": X},
                outputs={"myOutput": outputs})
        # Predictions
        y_test_pred = sess.run(outputs, feed_dict={X: x_test})

    # checking prediction output nos
    print(y_test_pred.shape)

    # ploting the graph
    # comp = pd.DataFrame({'Column1': y_test[:, 3], 'Column2': y_test_pred[:, 3]})
    # plt.figure(figsize=(10, 5))
    # plt.plot(comp['Column1'], color='blue', label='Target')
    # plt.plot(comp['Column2'], color='black', label='Prediction')
    # plt.legend()
    # plt.show()


if __name__=="__main__":

    abc = glob.glob(".\data_30\*.csv")
    data_dir = []

    for f in abc:
        data_dir.append(f.lstrip(".\\data_30\\").rstrip(".csv"))

    for in_file in data_dir:
        print("Processing :", in_file)
        main(in_file)
        time.sleep(60)
        # break
