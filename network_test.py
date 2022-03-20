import numpy as np
from network import Network, sigmoid, sigmoid_prime

import unittest, types


class TestNetwork(unittest.TestCase):
    def test_sigmoid(self):
        # Setup
        x = np.array([-2, -1, -0.5, -0.25, 0, .25, 0.5, 1, 2])
        expected = np.array([0.119202, 0.268941, 0.377541, 0.437823, 0.5,
                      0.562177, 0.622459, 0.731059, 0.880797])

        # SUT
        predicted = sigmoid(x)

        # Test
        np.testing.assert_allclose(predicted, expected, rtol=1e-05)


    def test_layer_sizes(self):
        # Setup
        expected_sizes = [3, 4, 2]
        expected_bias_shapes = [(size, ) for size in expected_sizes[1:]]
        expected_weight_shapes = [(r, c) for r, c in zip(expected_sizes[1:], expected_sizes[:-1])]

        # Sut
        n = Network(expected_sizes)

        # Test
        self.assertEqual(n.sizes, expected_sizes)
        self.assertEqual([b.shape for b in n.biases], expected_bias_shapes)
        self.assertEqual([w.shape for w in n.weights], expected_weight_shapes)


    def test_feed_forward(self):
        # Setup
        x = np.array([1, 2, 3])
        b2 = np.array([1, 2, 3, 4])
        b3 = np.array([1, 2])
        w2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        w3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        expected = sigmoid(w3 @ sigmoid(w2 @ x + b2) + b3)

        n = Network([3, 4, 2])
        n.biases = [b2, b3]
        n.weights = [w2, w3]

        # SUT
        predicted = n.feedforward(x)

        # Test
        np.testing.assert_allclose(predicted, expected)

    
    def test_feed_forward_intermediate(self):
        # Setup
        x = np.array([1, 2, 3])
        b2 = np.array([1, 2, 3, 4])
        b3 = np.array([1, 2])
        w2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        w3 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

        z2 = w2 @ x + b2
        a2 = sigmoid(z2)

        z3 = w3 @ a2 + b3
        a3 = sigmoid(z3)

        expected_weighted_inputs = [x, z2, z3]
        expected_activations = [x, a2, a3]

        n = Network([3, 4, 2])
        n.biases = [b2, b3]
        n.weights = [w2, w3]

        # SUT
        predicted_weighted_inputs, predicted_activations = n.feedforward_return_intermediate(x)

        # Test
        for p, e in zip(predicted_weighted_inputs, expected_weighted_inputs):
            np.testing.assert_allclose(p, e)

        for p, e in zip(predicted_activations, expected_activations):
            np.testing.assert_allclose(p, e)


    def test_convert_expected(self):
        # Setup
        n = Network([3, 4, 2])
        y = 1

        expected = np.array([0, 1])

        # SUT
        predicted = n.convert_expected(y)

        # Test
        np.testing.assert_array_equal(predicted, expected)


    def test_calculate_error(self):
        # Setup
        x = np.array(list(range(1, 301))) / 300

        n = Network([300, 40, 70, 20])
        y_converted = n.convert_expected(1)

        w3, w4 = n.weights[1], n.weights[2]

        z, a = n.feedforward_return_intermediate(x)
        (_, z2, z3, z4) = z
        a4 = a[-1]

        e4 = n.cost.delta(a4, y_converted, z4)
        e3 = (w4.transpose() @ e4) * sigmoid_prime(z3)
        e2 = (w3.transpose() @ e3) * sigmoid_prime(z2)
        e1 = np.zeros(x.shape)

        expected_errors = [e1, e2, e3, e4]

        # SUT
        predicted_errors = n.calculate_errors(z, a, y_converted)

        # Test
        for p, e in zip(predicted_errors, expected_errors):
            np.testing.assert_allclose(p, e)


    def test_backprop(self):
        # Setup
        x = np.array(list(range(1, 301))) / 300
        y = 1
        y_converted = np.zeros(20)
        y_converted[y] = 1

        n = Network([300, 40, 70, 20])

        z, a = n.feedforward_return_intermediate(x)
        e = n.calculate_errors(z, a, y_converted)

        a1, a2, a3, a4 = a
        _, e2, e3, e4 = e

        expected_nabla_b = [e2, e3, e4]
        expected_nabla_w = [np.outer(e2, a1), np.outer(e3, a2), np.outer(e4, a3)]

        # SUT
        predicted_nabla_b, predicted_nabla_w = n.backprop(x, y)

        # Test
        for p, e in zip(predicted_nabla_b, expected_nabla_b):
            np.testing.assert_allclose(p, e)

        for p, e in zip(predicted_nabla_w, expected_nabla_w):
            np.testing.assert_allclose(p, e)


    def test_train(self):
        # From: https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        # Setup, this example uses some different conventions
        # Namely, a different expected vector
        # And a vector cost function (which is wrong, but this is the best example I could find)
        def convert_expected(self, y):
            return np.array([0.01, 0.99])

        class ComponentWiseQuadraticCost:
            @staticmethod
            def delta(predicted, expected, weighted_input):
                return (predicted - expected) * sigmoid_prime(weighted_input)

        x = np.array([0.05, 0.10])
        y = 1
        eta = 0.5

        b2 = np.array([0.35, 0.35])
        b3 = np.array([0.60, 0.60])

        w2 = np.array([[0.15, 0.20], [0.25, 0.30]])
        w3 = np.array([[0.40, 0.45], [0.50, 0.55]])

        n = Network([2, 2, 2], cost = ComponentWiseQuadraticCost)
        n.convert_expected = types.MethodType(convert_expected, n)
        n.biases = [b2, b3]
        n.weights = [w2, w3]

        expected_w2 = np.array([
            [0.149780716, 0.19956143],
            [0.24975114, 0.29950229]
        ])

        expected_w3 = np.array([
            [0.35891648, 0.408666186],
            [0.511301270, 0.561370121]
        ])

        expected_activations = [
            np.array([0.05, 0.10]),
            np.array([0.593269992, 0.596884378]),
            np.array([0.75136507, 0.772928465])
        ]

        # SUT
        _, predicted_activations = n.feedforward_return_intermediate(x)
        n.train([[x, y]], eta)

        predicted_w2, predicted_w3 = n.weights

        # Test
        for p, e in zip(predicted_activations, expected_activations):
            np.testing.assert_allclose(p, e)
            
        np.testing.assert_allclose(predicted_w2, expected_w2)
        np.testing.assert_allclose(predicted_w3, expected_w3)
        




        






if __name__ == "__main__":
    unittest.main()