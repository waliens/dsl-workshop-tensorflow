try:
    import tensorflow as tf
    from tensorflow.contrib.learn.python.learn import datasets
except ImportError:
    print("TensorFlow seems to be missing...")
    exit(1)

try:
    import keras
except ImportError:
    print("Keras seems to be missing...")
    exit(1)

if __name__ == "__main__":
    print("Required libraries seems to have been installed ! Good job !")

    try:
        mnist = datasets.mnist.load_mnist("./data")
        x_train = mnist.train.images
        y_train = mnist.train.labels
        x_val = mnist.validation.images
        y_val = mnist.validation.labels
        x_test = mnist.test.images
        y_test = mnist.test.labels
        print("Downloaded {} images successfully...".format(x_train.shape[0] + x_val.shape[0] + x_test.shape[0]))
    except Exception as e:
        print("Dataset couldn't be downloaded: {}".format(str(e)))
        exit(1)