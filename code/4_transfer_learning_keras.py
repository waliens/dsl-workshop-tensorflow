import numpy as np

from keras import Model
from keras.applications import ResNet50
from keras.layers import Dense
from keras.optimizers import sgd
from sklearn.metrics import accuracy_score

from tensorflow.contrib.learn.python.learn import datasets


def build_model(height, width, n_classes):
    resnet = ResNet50(input_shape=(None, height, width, 1), weights="imagenet", include_top=False)
    for layer in resnet.layers:
        layer.trainable = False
    out = Dense(n_classes, activation="softmax")(resnet.output)
    return Model(inputs=resnet.input, outputs=out)


def main():
    epochs = 25
    batch_size = 430
    height, width = 28, 28
    n_classes = 10

    # build model
    model = build_model(height, width, n_classes)
    model.compile(sgd(5e-2), "categorical_crossentropy", metrics=["accuracy"])

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