import math
import numpy as np
import pandas
from sklearn.preprocessing import MinMaxScaler
import keras
import matplotlib.pyplot as plt
import requests
from datetime import datetime
import tensorflow as tf

'''
from keras.models import Sequential
from keras.layers import Dense, LSTM
'''

plt.style.use('fivethirtyeight')

# Get the stock quote
api_key = 'MOJKLPNZSRJ67EVD'
symbol = 'AAPL'

info = requests.get(
    'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=' + symbol + '&outputsize=full&apikey=' + api_key)

info = info.json()
close_dict = {'Date': [], 'Close': []}
for date in info['Time Series (Daily)']:
    close_dict['Date'].append(datetime.strptime(date, '%Y-%m-%d'))
    close_dict['Close'].append(float(info['Time Series (Daily)'][date]['4. close']))
# print(close_dict)


df = pandas.DataFrame(close_dict)
series = pandas.Series(data=close_dict, index=['Date', 'Close'])

print(df.info())

# Show the data
print(df)

# Get the number of rows and columns in the data set
df.shape

# visualize the closing price history
plt.figure(figsize=(16, 8))
plt.title('Close Price History')
plt.plot(df['Date'], df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD($)', fontsize=18)
# plt.show()

# create a new dataframe with only the 'Close' column
data = df.filter(['Close'])
# print("close data")
# print(data)

# Convert the dataframe to a numpy array
dataset = data.values

# get the number of rows to train the bodel on
training_data_len = math.ceil(len(dataset) * .8)
test_data_len = math.ceil(len(dataset) * .2)
print("training data length")
print(training_data_len)
print("test data length")
print(test_data_len)

# setting data
train_data = dataset[:training_data_len]
test_data = dataset[training_data_len:]

# Scale the data
scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(dataset)
# print("scaled data")
# print(scaled_data)

# Create the training data set
# create the scaled training dataset
# train_data = scaled_data[0:training_data_len, :]
train_data = train_data.reshape(-1, 1)
test_data = test_data.reshape(-1, 1)

# Train the Scaler with training data and smooth data
smoothing_window_size = (math.ceil(training_data_len / 4))
for di in range(0, 1000,
                smoothing_window_size):  # 1000 made it work, 10000 and len(dataset) broke it. not optimized idk what this is yet
    scaler.fit(train_data[di:di + smoothing_window_size, :])
    train_data[di:di + smoothing_window_size, :] = scaler.transform(train_data[di:di + smoothing_window_size, :])

# You normalize the last bit of remaining data
scaler.fit(train_data[di + smoothing_window_size:, :])
train_data[di + smoothing_window_size:, :] = scaler.transform(train_data[di + smoothing_window_size:, :])

# Reshape both train and test data
train_data = train_data.reshape(-1)

# Normalize test data
test_data = scaler.transform(test_data).reshape(-1)

# Note that you should only smooth training data.
# Now perform exponential moving average smoothing
# So the data will have a smoother curve than the original ragged data
EMA = 0.0
gamma = 0.1
for ti in range(training_data_len):
    EMA = gamma * train_data[ti] + (1 - gamma) * EMA
    train_data[ti] = EMA

# Used for visualization and test purposes
all_mid_data = np.concatenate([train_data, test_data], axis=0)


class DataGeneratorSeq(object):

    def __init__(self, prices, batch_size, num_unroll):
        self._prices = prices
        self._prices_length = len(self._prices) - num_unroll
        self._batch_size = batch_size
        self._num_unroll = num_unroll
        self._segments = self._prices_length // self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def next_batch(self):

        batch_data = np.zeros((self._batch_size), dtype=np.float32)
        batch_labels = np.zeros((self._batch_size), dtype=np.float32)

        for b in range(self._batch_size):
            if self._cursor[b] + 1 >= self._prices_length:
                # self._cursor[b] = b * self._segments
                self._cursor[b] = np.random.randint(0, (b + 1) * self._segments)

            batch_data[b] = self._prices[self._cursor[b]]
            batch_labels[b] = self._prices[self._cursor[b] + np.random.randint(0, 5)]

            self._cursor[b] = (self._cursor[b] + 1) % self._prices_length

        return batch_data, batch_labels

    def unroll_batches(self):

        unroll_data, unroll_labels = [], []
        init_data, init_label = None, None
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()

            unroll_data.append(data)
            unroll_labels.append(labels)

        return unroll_data, unroll_labels

    def reset_indices(self):
        for b in range(self._batch_size):
            self._cursor[b] = np.random.randint(0, min((b + 1) * self._segments, self._prices_length - 1))


dg = DataGeneratorSeq(train_data, 5, 5)
u_data, u_labels = dg.unroll_batches()

for ui, (dat, lbl) in enumerate(zip(u_data, u_labels)):
    print('\n\nUnrolled index %d' % ui)
    dat_ind = dat
    lbl_ind = lbl
    print('\tInputs: ', dat)
    print('\n\tOutput:', lbl)

D = 1  # Dimensionality of the data. Since your data is 1-D this would be 1
num_unrollings = 50  # Number of time steps you look into the future.
batch_size = 500  # Number of samples in a batch
num_nodes = [200, 200, 150]  # Number of hidden nodes in each layer of the deep LSTM stack we're using
n_layers = len(num_nodes)  # number of layers
dropout = 0.2  # dropout amount

# tf.reset_default_graph() # This is important in case you run this multiple times ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Input data.
train_inputs, train_outputs = [], []

# You unroll the input over time defining placeholders for each time step
for ui in range(num_unrollings):
    train_inputs.append(tf.keras.Input(dtype=tf.float32, shape=[batch_size, D], name='train_inputs_%d' % ui))
    train_outputs.append(tf.keras.Input(dtype=tf.float32, shape=[batch_size, 1], name='train_outputs_%d' % ui))

lstm_cells = [
    tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=num_nodes[li],
                            state_is_tuple=True,
                            initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform")
                            )
    for li in range(n_layers)]

drop_lstm_cells = [tf.nn.RNNCellDropoutWrapper(
    lstm, input_keep_prob=1.0, output_keep_prob=1.0 - dropout, state_keep_prob=1.0 - dropout
) for lstm in lstm_cells]
drop_multi_cell = tf.keras.layers.StackedRNNCells(drop_lstm_cells)  #tf.nn.rnn_cell.MultiRNNCell>replaced in tf 2.0> tf.keras.layers.StackedRNNCells but are they the same?
multi_cell = tf.keras.layers.StackedRNNCells(lstm_cells)

w = tf.compat.v1.get_variable('w', shape=[num_nodes[-1], 1], initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
b = tf.compat.v1.get_variable('b', initializer=tf.random.uniform([1], -0.1, 0.1))

# Create cell state and hidden state variables to maintain the state of the LSTM
c, h = [], []
initial_state = []
for li in range(n_layers):
    c.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    h.append(tf.Variable(tf.zeros([batch_size, num_nodes[li]]), trainable=False))
    initial_state.append(tf.compat.v1.nn.rnn_cell.LSTMStateTuple(c[li], h[li])) #answer found on google
                                                                #tf.nn.rnn_cell.LSTMStateTuple>changed in 2.0> tf.keras.layers.LSTMCell
                                                                #tensorflow.python.keras.layers.legacy_rnn.rnn_cell_impl.LSTMCell >not optimized for gpu preformance> tf.contrib.cudnn_rnn.CudnnLSTM
                                                                #layer.add_variable> removed, use> layer.add_weight


# Do several tensor transformations, because the function dynamic_rnn requires the output to be of
# a specific format. Read more at: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
all_inputs = tf.concat([tf.expand_dims(t, 0) for t in train_inputs], axis=0)
print(initial_state)
print(all_inputs)
# all_outputs is [seq_length, batch_size, num_nodes]
all_lstm_outputs, state = tf.keras.layers.RNN( #tf.compat.v1.nn.dynamic_rnn>updated to > tf.keras.layers.RNN
    cell=drop_multi_cell, inputs=all_inputs, initial_state=tuple(initial_state),
    time_major=True, dtype=tf.float32, **kwargs) #**kwargs)

all_lstm_outputs = tf.reshape(all_lstm_outputs, [batch_size * num_unrollings, num_nodes[-1]])

all_outputs = tf.compat.v1.nn.xw_plus_b(all_lstm_outputs, w, b)

split_outputs = tf.split(all_outputs, num_unrollings, axis=0)

# When calculating the loss you need to be careful about the exact form, because you calculate
# loss of all the unrolled steps at the same time
# Therefore, take the mean error or each batch and get the sum of that over all the unrolled steps

print('Defining training Loss')
loss = 0.0
with tf.control_dependencies([tf.compat.v1.assign(c[li], state[li][0]) for li in range(n_layers)] +
                             [tf.compat.v1.assign(h[li], state[li][1]) for li in range(n_layers)]):
    for ui in range(num_unrollings):
        loss += tf.reduce_mean(input_tensor=0.5 * (split_outputs[ui] - train_outputs[ui]) ** 2)

print('Learning rate decay operations')
global_step = tf.Variable(0, trainable=False)
inc_gstep = tf.compat.v1.assign(global_step, global_step + 1)
tf_learning_rate = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)
tf_min_learning_rate = tf.compat.v1.placeholder(shape=None, dtype=tf.float32)

learning_rate = tf.maximum(
    tf.compat.v1.train.exponential_decay(tf_learning_rate, global_step, decay_steps=1, decay_rate=0.5, staircase=True),
    tf_min_learning_rate)

# Optimizer.
print('TF Optimization operations')
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
gradients, v = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimizer = optimizer.apply_gradients(
    zip(gradients, v))

print('\tAll done')

epochs = 30
valid_summary = 1 # Interval you make test predictions

n_predict_once = 50 # Number of steps you continously predict for

train_seq_length = train_data.size # Full length of the training data

train_mse_ot = [] # Accumulate Train losses
test_mse_ot = [] # Accumulate Test loss
predictions_over_time = [] # Accumulate predictions

session = tf.compat.v1.InteractiveSession()

tf.compat.v1.global_variables_initializer().run()

# Used for decaying learning rate
loss_nondecrease_count = 0
loss_nondecrease_threshold = 2 # If the test error hasn't increased in this many steps, decrease learning rate

print('Initialized')
average_loss = 0

# Define data generator
data_gen = DataGeneratorSeq(train_data,batch_size,num_unrollings)

x_axis_seq = []

# Points you start your test predictions from
test_points_seq = np.arange(11000,12000,50).tolist()

for ep in range(epochs):

    # ========================= Training =====================================
    for step in range(train_seq_length//batch_size):

        u_data, u_labels = data_gen.unroll_batches()

        feed_dict = {}
        for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):
            feed_dict[train_inputs[ui]] = dat.reshape(-1,1)
            feed_dict[train_outputs[ui]] = lbl.reshape(-1,1)

        feed_dict.update({tf_learning_rate: 0.0001, tf_min_learning_rate:0.000001})

        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

    # ============================ Validation ==============================
    if (ep+1) % valid_summary == 0:

      average_loss = average_loss/(valid_summary*(train_seq_length//batch_size))

      # The average loss
      if (ep+1)%valid_summary==0:
        print('Average loss at step %d: %f' % (ep+1, average_loss))

      train_mse_ot.append(average_loss)

      average_loss = 0 # reset loss

      predictions_seq = []

      mse_test_loss_seq = []

      # ===================== Updating State and Making Predicitons ========================
      for w_i in test_points_seq:
        mse_test_loss = 0.0
        our_predictions = []

        if (ep+1)-valid_summary==0:
          # Only calculate x_axis values in the first validation epoch
          x_axis=[]

        # Feed in the recent past behavior of stock prices
        # to make predictions from that point onwards
        for tr_i in range(w_i-num_unrollings+1,w_i-1):
          current_price = all_mid_data[tr_i]
          feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)
          _ = session.run(sample_prediction,feed_dict=feed_dict)

        feed_dict = {}

        current_price = all_mid_data[w_i-1]

        feed_dict[sample_inputs] = np.array(current_price).reshape(1,1)

        # Make predictions for this many steps
        # Each prediction uses previous prediciton as it's current input
        for pred_i in range(n_predict_once):

          pred = session.run(sample_prediction,feed_dict=feed_dict)

          our_predictions.append(np.asscalar(pred))

          feed_dict[sample_inputs] = np.asarray(pred).reshape(-1,1)

          if (ep+1)-valid_summary==0:
            # Only calculate x_axis values in the first validation epoch
            x_axis.append(w_i+pred_i)

          mse_test_loss += 0.5*(pred-all_mid_data[w_i+pred_i])**2

        session.run(reset_sample_states)

        predictions_seq.append(np.array(our_predictions))

        mse_test_loss /= n_predict_once
        mse_test_loss_seq.append(mse_test_loss)

        if (ep+1)-valid_summary==0:
          x_axis_seq.append(x_axis)

      current_test_mse = np.mean(mse_test_loss_seq)

      # Learning rate decay logic
      if len(test_mse_ot)>0 and current_test_mse > min(test_mse_ot):
          loss_nondecrease_count += 1
      else:
          loss_nondecrease_count = 0

      if loss_nondecrease_count > loss_nondecrease_threshold :
            session.run(inc_gstep)
            loss_nondecrease_count = 0
            print('\tDecreasing learning rate by 0.5')

      test_mse_ot.append(current_test_mse)
      print('\tTest MSE: %.5f'%np.mean(mse_test_loss_seq))
      predictions_over_time.append(predictions_seq)
      print('\tFinished Predictions')

best_prediction_epoch = 28 # replace this with the epoch that you got the best results when running the plotting code

plt.figure(figsize = (18,18))
plt.subplot(2,1,1)
plt.plot(range(df.shape[0]),all_mid_data,color='b')

# Plotting how the predictions change over time
# Plot older predictions with low alpha and newer predictions with high alpha
start_alpha = 0.25
alpha  = np.arange(start_alpha,1.1,(1.0-start_alpha)/len(predictions_over_time[::3]))
for p_i,p in enumerate(predictions_over_time[::3]):
    for xval,yval in zip(x_axis_seq,p):
        plt.plot(xval,yval,color='r',alpha=alpha[p_i])

plt.title('Evolution of Test Predictions Over Time',fontsize=18)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.xlim(11000,12500)

plt.subplot(2,1,2)

# Predicting the best test prediction you got
plt.plot(range(df.shape[0]),all_mid_data,color='b')
for xval,yval in zip(x_axis_seq,predictions_over_time[best_prediction_epoch]):
    plt.plot(xval,yval,color='r')

plt.title('Best Test Predictions Over Time',fontsize=18)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.xlim(11000,12500)
plt.show()
