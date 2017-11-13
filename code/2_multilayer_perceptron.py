import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.learn.python.learn import datasets


def layer(input_layer, input_size, output_size, keep_proba=None, name=""):
    with tf.variable_scope(name):
        w = tf.Variable(
            initial_value=tf.truncated_normal([input_size, output_size], seed=np.random.randint(9999999)),
            trainable=True,
            name="weights"
        )
        bias = tf.Variable(
            initial_value=tf.zeros([output_size]),
            trainable=True,
            name="bias"
        )

        # compute activation
        prod = tf.matmul(input_layer, w)
        with_bias = tf.nn.bias_add(prod, bias)
        activation = tf.nn.sigmoid(with_bias, name="out")

        if keep_proba is not None:
            return tf.nn.dropout(activation, keep_prob=keep_proba, name="dropout")
        else:
            return activation


def build_model(hidden_layers, n_classes, n_inputs=784, batch_size=None):
    # building the graph
    x = tf.placeholder(shape=[batch_size, n_inputs], dtype=tf.float32, name="x")
    y_true = tf.placeholder(shape=[batch_size, n_classes], dtype=tf.float32, name="y_true")
    keep_proba = tf.placeholder(shape=[], dtype=tf.float32, name="keep_proba")

    # build layers
    prev_layer = x
    prev_size = n_inputs
    for i, size in enumerate(hidden_layers):
        prev_layer = layer(
            input_layer=prev_layer,
            input_size=prev_size,
            output_size=size,
            keep_proba=keep_proba,
            name="hidden_{}".format(i + 1)
        )
        prev_size = size

    # classification layer
    classif = layer(
        input_layer=prev_layer,
        input_size=prev_size,
        output_size=n_classes,
        name="classif_layer"
    )

    return x, y_true, tf.nn.softmax(classif), keep_proba


def cross_entropy(y_true, y_pred):
    with tf.variable_scope("cross_entropy"):
        per_sample_loss = tf.reduce_sum(y_true * tf.log(y_pred + 1e-8), axis=-1)
        return - tf.reduce_mean(per_sample_loss)


def evaluate_model(sess, x, y, x_test, y_test, batch_size=64, other_feed=None):
    other_feed = dict() if other_feed is None else other_feed
    n_samples = x_test.shape[0]
    n_batches = int(math.ceil(n_samples / batch_size))
    y_pred = np.zeros([n_samples], dtype=np.float)
    for i in range(n_batches):
        start = i * batch_size
        end = min(n_samples, start + batch_size)
        _y, = sess.run([y], feed_dict={x: x_test[start:end], **other_feed})
        y_pred[start:end] = np.argmax(_y, axis=-1)
    acc = accuracy_score(np.argmax(y_test, axis=-1), y_pred)
    return acc


def main():
    np.random.seed(42)

    # hyper-parameters
    batch_size = 128
    epochs = 200
    iter_per_epoch = 400
    learning_rate = 5e-2
    hidden_layers = [64, 64, 32, 8]
    n_classes = 10
    drop_prob = 0.0

    # build graph and training machinery
    x, y_true, y_pred, keep_proba = build_model(
        hidden_layers,
        n_classes=n_classes,
        n_inputs=784
    )
    loss = cross_entropy(y_true, y_pred)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    # get data
    mnist = datasets.mnist.read_data_sets("../data", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_val = mnist.validation.images
    y_val = mnist.validation.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

    # build an initializer for all variables
    initializer = tf.global_variables_initializer()

    # optimize
    with tf.Session() as sess:
        # to get reproducible results
        tf.set_random_seed(42)

        # save graph for TensorBoard
        writer = tf.summary.FileWriter("./logs", sess.graph)

        # initialize the variables
        sess.run([initializer])

        print("Train")
        for i in range(epochs):
            losses = list()
            for j in range(iter_per_epoch):
                # select a random subset of training data
                idx = np.random.choice(x_train.shape[0], batch_size)

                # optimize = run the optimizer with correct inputs
                feed = {
                    x: x_train[idx, :],
                    y_true: y_train[idx],
                    keep_proba: (1 - drop_prob)
                }

                _loss, _ = sess.run([loss, optimizer], feed_dict=feed)
                losses.append(_loss)

            val_acc = evaluate_model(sess, x, y_pred, x_val, y_val, batch_size=128, other_feed={keep_proba: 1.0})
            print("> #{: <5} train_loss:{:.4f} val_acc:{:.4f}".format(i, np.mean(losses[-10:]), val_acc))

    print("Test")
    test_acc = evaluate_model(sess, x, y_pred, x_test, y_test)
    print("> accuracy: {}".format(test_acc))


if __name__ == "__main__":
    main()