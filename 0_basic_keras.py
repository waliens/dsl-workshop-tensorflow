from math import pi
import numpy as np
from keras import backend as K, Model
from keras import Input
from keras.layers import Add, Lambda

if __name__ == "__main__":
    # graph definition
    a = Input(batch_shape=(1,), dtype=K.floatx())
    b = Input(batch_shape=(1,), dtype=K.floatx())
    radicand = Add()([a, Lambda(lambda x: K.sin(x))(b)])
    y = Lambda(lambda x: K.sqrt(x))(radicand)
    model = Model(inputs=[a, b], outputs=[y])

    in_a = np.array([3])
    in_b = np.array([pi / 2.0])
    outputs = model.predict([in_a, in_b])

    print(outputs)