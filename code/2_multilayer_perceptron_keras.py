import numpy as np
from keras import Input
from keras.engine import Model
from keras.layers import Dense, Dropout
from keras.optimizers import sgd
from sklearn.metrics import accuracy_score
from tensorflow.contrib.learn.python.learn import datasets


def build_model(hidden_layers, n_classes, n_inputs=784, keep_proba=0.5):
    input = Input(shape=(n_inputs,))

    # create layers
    x = input
    for size in hidden_layers:
        x = Dense(size, activation="sigmoid", use_bias=True)(x)
        x = Dropout(rate=keep_proba)(x)

    # classification layer
    x = Dense(n_classes, activation="softmax")(x)
    return Model(inputs=[input], outputs=[x])


if __name__ == "__main__":
    # hyper-parameters
    batch_size = 128
    epochs = 200
    learning_rate = 5e-2
    hidden_layers = [64, 64, 32, 8]
    n_classes = 10
    keep_proba = 0.0

    # model
    model = build_model(hidden_layers, n_classes, keep_proba=keep_proba)
    model.compile(sgd(lr=5e-2), "categorical_crossentropy", metrics=["accuracy"])

    # get data
    mnist = datasets.mnist.read_data_sets("./data", one_hot=True)
    x_train = mnist.train.images
    y_train = mnist.train.labels
    x_val = mnist.validation.images
    y_val = mnist.validation.labels
    x_test = mnist.test.images
    y_test = mnist.test.labels

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
    acc = accuracy_score(np.argmax(y_test, axis=-1), np.argmax(y_pred, axis=-1))
    print("> accuracy: {}".format(acc))