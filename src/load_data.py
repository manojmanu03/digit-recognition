from tensorflow.keras.datasets import mnist
import numpy as np

def get_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Normalize and reshape for CNN
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Reshape for Conv2D: (batch_size, height, width, channels)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, y_train, X_test, y_test
