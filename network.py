# Based on: http://neuralnetworksanddeeplearning.com/chap1.html

import random

import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.layer_count = len(self.sizes)

        self.biases = [np.random.randn(s) for s in sizes[1:]]
        self.weights = [np.random.randn(s1, s2) for s1, s2 in zip(sizes[1:], sizes[:-1])]

    
    def stochastic_gradient_descent(self, training_data, epochs, batch_size, eta, test_data = None):
        n = len(training_data)

        if test_data:
            n_tests = len(test_data)

        for i in range(epochs):
            random.shuffle(training_data)

            batches = [training_data[b : b + batch_size] for b in range(0, n, batch_size)]

            for batch in batches:
                self.train(batch, eta)

            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {n_tests}")

            else:
                print(f"Epoch {i} complete.")


    def feedforward(self, x):
        out = x

        for weight_matrix, bias in zip(self.weights, self.biases):
            out = sigmoid(weight_matrix @ out + bias)

        return out

    
    # used in backprop
    def feedforward_return_intermediate(self, x):
        activations = [np.zeros(size) for size in self.sizes]
        weighted_inputs = [np.zeros(size) for size in self.sizes]

        activations[0] = x
        weighted_inputs[0] = x

        for i in range(self.layer_count - 1):
            w = self.weights[i]
            b = self.biases[i]
            a = activations[i]

            weighted_inputs[i + 1] = w @ a + b
            activations[i + 1] = sigmoid(weighted_inputs[i + 1])

        return weighted_inputs, activations


    def predict(self, x):
        y = self.feedforward(x)

        return np.argmax(y)


    def evaluate(self, test_data):
        return sum(self.predict(x) == y for x, y in test_data)


    def convert_expected(self, y):
        expected = np.zeros(self.sizes[-1])
        expected[y] = 1

        return expected


    def backprop(self, x, y):
        expected = self.convert_expected(y)
        weighted_inputs, activations = self.feedforward_return_intermediate(x)
        errors = self.calculate_errors(weighted_inputs, activations, expected)

        nabla_b = [np.array(error) for error in errors[1:]]
        nabla_w = [np.outer(e, a) for e, a in zip(errors[1:], activations[:-1])]

        return nabla_b, nabla_w


    def calculate_errors(self, weighted_inputs, activations, expected):
        errors = [np.zeros(size) for size in self.sizes]
        predicted = activations[-1]

        errors[-1] = self.cost_derivative(predicted, expected) * sigmoid_prime(weighted_inputs[-1])

        # The final error is excluded because it is not needed
        for i in range(self.layer_count-2, 0, -1):
            error_front = errors[i + 1]
            w_front = self.weights[i] # Would be i+1, but weights only exist between layers, so there is one less
            z_back = weighted_inputs[i]

            errors[i] = np.transpose(w_front) @ error_front * sigmoid_prime(z_back)

        return errors


    def cost_derivative(self, predicted, expected):
        return np.sum(predicted - expected)


    def train(self, data, eta):
        m = len(data)

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in data:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.biases = [b - (eta / m) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [w - (eta / m) * nw for w, nw in zip(self.weights, nabla_w)]