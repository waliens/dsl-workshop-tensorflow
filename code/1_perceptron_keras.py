import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense, Dropout
from keras.optimizers import sgd
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.contrib.learn.python.learn import datasets


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


def build_model(n_inputs=784):
    input = Input(shape=(n_inputs,))
    x = Dense(1, activation="sigmoid", kernel_initializer="TruncatedNormal", use_bias=True)(input)
    return Model(inputs=[input], outputs=[x])


if __name__ == "__main__":
    # hyper-parameters
    batch_size = 128
    epochs = 10
    learning_rate = 5e-2

    # model
    model = build_model()
    model.compile(sgd(lr=5e-2), "binary_crossentropy", metrics=["accuracy"])

    # get data
    x_train, y_train, x_val, y_val, x_test, y_test = extract_two_digits(3, 8)

    # train
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=2
    )

    # evaluate on test set
    y_pred = model.predict(x_test, batch_size=batch_size)
    print("Test")
    acc = accuracy_score(y_test, (y_pred > 0.5).astype(np.int))
    roc = roc_auc_score(y_test, y_pred)
    print("> accuracy: {}".format(acc))
    print("> roc_auc : {}".format(roc))