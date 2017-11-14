import numpy as np
from sklearn.metrics import accuracy_score

from tensorflow.contrib.learn.python.learn import datasets

from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras.optimizers import sgd


def build_model(height, width, n_classes):
    input = Input(shape=[height, width, 1])

    # layer 1
    x = Conv2D(32, kernel_size=3, padding="same")(input)
    x = Activation("relu")(x)

    # layer 2
    x = Conv2D(32, kernel_size=3, padding="same")(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
    x = Activation("relu")(x)

    # layer 3
    x = Conv2D(16, kernel_size=3, padding="same")(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding="same")(x)
    x = Activation("relu")(x)

    # fully connected
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(n_classes, activation="softmax")(x)

    return Model(inputs=[input], outputs=[x])


def main():
    epochs = 25
    batch_size = 430
    height, width = 28, 28
    n_classes = 10

    model = build_model(height, width, n_classes)
    model.compile(sgd(lr=5e-2), "categorical_crossentropy", metrics=["accuracy"])

    # get data
    mnist = datasets.mnist.read_data_sets("../data", one_hot=True)
    x_train = np.reshape(mnist.train.images, (-1, height, width, 1))
    y_train = mnist.train.labels
    x_val = np.reshape(mnist.validation.images, (-1, height, width, 1))
    y_val = mnist.validation.labels
    x_test = np.reshape(mnist.test.images, (-1, height, width, 1))
    y_test = mnist.test.labels

    # train
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # evaluate on test set
    y_pred = model.predict(x_test, batch_size=batch_size)
    print("Test")
    acc = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1))
    print("> accuracy: {}".format(acc))


if __name__ == "__main__":
    main()