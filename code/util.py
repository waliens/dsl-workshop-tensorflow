import numpy as np
import math

from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.contrib.learn.python.learn import datasets


def extract_two_digits(digit0, digit1):
    # load data
    mnist = datasets.mnist.load_mnist("../data")
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


def evaluate_binary_model(sess, x, y, x_test, y_test, batch_size=64):
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