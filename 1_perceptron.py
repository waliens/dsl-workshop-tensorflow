import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn import datasets


def build_model(n_inputs=784, batch_size=None):
    # building the graph
    x = tf.placeholder(shape=[batch_size, n_inputs], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[batch_size], dtype=tf.float32, name="y")

    w = tf.Variable(
        initial_value=tf.truncated_normal(shape=[n_inputs, 1]),
        trainable=True,
        name="w"
    )
    b = tf.Variable(
        initial_value=tf.zeros(shape=[1], dtype=tf.float32),
        trainable=True,
        name="b"
    )

    a = tf.nn.bias_add(tf.matmul(x, w), b)
    return x, y, tf.nn.sigmoid(a, name="activation"),


def extract_two_digits(digit0, digit1):
    # load data
    mnist = datasets.mnist.load_mnist("./data")
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_val = mnist.validation.images
    y_val = mnist.validation.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    # look for the digits to extract
    digits = [digit0, digit1]
    train_idx = np.in1d(y_train, digits)
    val_idx = np.in1d(y_val, digits)
    test_idx = np.in1d(y_test, digits)

    # actually select two digits
    x_train, x_val, x_test = x_train[train_idx], x_val[val_idx], x_test[test_idx]
    y_train, y_val, y_test = y_train[train_idx], y_val[val_idx], y_test[test_idx]

    # cast to a binary problem
    y_train[y_train == digit0] = 0
    y_train[y_train == digit1] = 1
    y_val[y_val == digit0] = 0
    y_val[y_val == digit1] = 1
    y_test[y_test == digit0] = 0
    y_test[y_test == digit1] = 1

    return x_train, y_train, x_val, y_val, x_test, y_test


if __name__ == "__main__":
    # hyperparameters
    batch_size = 64
    iterations = 100000

    # build graph and training machinery
    x, y, y_pred = build_model()
    loss = - tf.reduce_mean(y * tf.log(y_pred) + (1 - y) * (1 - tf.log(y_pred)), name="loss")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    # get data
    x_train, y_train, x_val, y_val, x_test, y_test = extract_two_digits(3, 9)

    # build an initializer for all variables
    initializer = tf.initialize_all_variables()

    # optimize
    with tf.Session() as sess:
        # save graph for TensorBoard
        writer = tf.summary.FileWriter("./logs", sess.graph)

        # initialize the variables
        sess.run([initializer])

        for i in range(iterations):  # 1000 iterations
            # select a random subset of training data
            idx = np.random.choice(x_train.shape[0], batch_size)

            # optimize = run the optimizer with correct inputs
            feed = {
                x: x_train[idx, :],
                y: y_train[idx]
            }
            _loss, _ = sess.run([loss, optimizer], feed_dict=feed)

            print("#{: <5} loss:{}".format(i, _loss))
