from keras import Input
from keras.engine import Model
from keras.layers import Conv2D, Activation, MaxPooling2D, Dense


def build_model(height, width, n_classes):
    input = Input(shape=[height, width])

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
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(n_classes, activation="softmax")(x)

    return Model(inputs=[input], outputs=[x])


def main():
    pass


if __name__ == "__main__":
    main()