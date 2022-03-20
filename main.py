import network

from tensorflow.keras.datasets import mnist

import numpy as np


def convert_data(data, size):
    return [(x.flatten() / 255.0, y) for x, y in zip(data[0][:size], data[1][:size])]



if __name__ == "__main__":
    training_data_raw, test_data_raw = mnist.load_data()

    print("Converting data")
    training_data = convert_data(training_data_raw, 60000)
    test_data = convert_data(test_data_raw, 10000)

    print("Starting learning")
    n = network.Network([784, 30, 10], cost = network.CrossEntropyCost)
    n.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data)