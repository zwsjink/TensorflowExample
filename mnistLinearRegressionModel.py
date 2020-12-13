import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical


def batch_iter(sourceData, batch_size, num_epochs, shuffle=True):
    data = np.array(sourceData)  # 将sourceData转换为array存储
    data_size = len(sourceData)
    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = sourceData[shuffle_indices]
        else:
            shuffled_data = sourceData

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)

            yield shuffled_data[start_index:end_index]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_reshape = x_train.reshape((-1, 28*28))
y_train_reshape=to_categorical(y_train)
print(x_train.shape)


learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
#
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax
#
# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(x_train.shape[0]/batch_size)
        # Loop over all batches
        # print(total_batch)
        x_iter = batch_iter(x_train_reshape, batch_size, total_batch, shuffle=True)
        y_iter = batch_iter(y_train_reshape, batch_size, total_batch, shuffle=True)
        for i in range(total_batch):
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs, batch_ys = [next(x_iter), next(y_iter)]
    #
    #
    #         # Run optimization op (backprop) and cost op (to get loss value)
    #         _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs, y: batch_ys})

            sess.run(tf.train.GradientDescentOptimizer(0.01).minimize(cost), feed_dict={x: batch_xs, y: batch_ys})
    print(sess.run(W))
    #         # Compute average loss
    #         avg_cost += c / total_batch
    #     # Display logs per epoch step
    #     if (epoch+1) % display_step == 0:
    #         print(cost)
    #         # print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    #
    # print("Optimization Finished!")

    # # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy:", accuracy.eval({x:x_test , y: y_test}))