import tensorflow as tf
import numpy as np
import math

from sklearn.metrics import roc_auc_score, accuracy_score
from tensorflow.contrib.learn.python.learn import datasets


def build_model(n_inputs=784, batch_size=None):
    # building the graph
    x = tf.placeholder(shape=[batch_size, n_inputs], dtype=tf.float32, name="x")
    y_true = tf.placeholder(shape=[batch_size], dtype=tf.float32, name="y_true")

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
    return x, y_true, tf.nn.sigmoid(a, name="activation")


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


def evaluate_model(sess, x, y, x_test, y_test, batch_size=64):
    n_samples = x_test.shape[0]
    n_batches = int(math.ceil(n_samples / batch_size))
    y_pred = np.zeros([n_samples], dtype=np.float)
    for i in range(n_batches):
        start = i * batch_size
        end = min(n_samples, start + batch_size)
        _y, = sess.run([y], feed_dict={x: x_test[start:end]})
        y_pred[start:end] = np.squeeze(_y)
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(np.int))
    roc = roc_auc_score(y_test, y_pred)
    return acc, roc


def main():
    # hyper-parameters
    batch_size = 128
    epochs = 100
    iter_per_epoch = 1000
    learning_rate = 1e-3

    # build graph and training machinery
    x, y_true, y_pred = build_model()
    loss = - tf.reduce_mean(y_true * tf.log(y_pred + 1e-8) + (1 - y_true) * tf.log(1 - y_pred + 1e-8), name="loss")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # get data
    x_train, y_train, x_val, y_val, x_test, y_test = extract_two_digits(0, 1)

    # build an initializer for all variables
    initializer = tf.global_variables_initializer()

    # optimize
    with tf.Session() as sess:
        # to get reproducible results
        tf.set_random_seed(42)
        np.random.seed(42)

        # save graph for TensorBoard
        writer = tf.summary.FileWriter("./logs", sess.graph)

        # initialize the variables
        sess.run([initializer])

        print("Train")
        for i in range(epochs):
            for j in range(iter_per_epoch):  # 1000 iterations
                # select a random subset of training data
                idx = np.random.choice(x_train.shape[0], batch_size)

                # optimize = run the optimizer with correct inputs
                feed = {
                    x: x_train[idx, :],
                    y_true: y_train[idx]
                }
                _loss, _ = sess.run([loss, optimizer], feed_dict=feed)

            val_acc, val_roc = evaluate_model(sess, x, y_pred, x_val, y_val, batch_size=128)
            print("> #{: <5} train_loss:{:.4f} val_acc:{:.4f} val_roc:{:.4f}".format(i, _loss, val_acc, val_roc))

        print("Test")
        test_acc, test_roc = evaluate_model(sess, x, y_pred, x_test, y_test)
        print("> accuracy: {}".format(test_acc))
        print("> roc_auc : {}".format(test_roc))


if __name__ == "__main__":
    main()