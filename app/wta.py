import numpy as np
import random
import math
from functools import reduce

class WTA:
    weights = []
    biases = []
    eta = 0.3
    deta = 0.98

    def __init__(self, num_neurons, num_inputs, numSteps):
        for i in range(num_neurons):
            neuron_weights = [random.random() for i in range(num_inputs)]
            self.weights.append(neuron_weights)
        self.deta = self.eta / (numSteps + 1 )

    def activation(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self, input):
        network_errors = 0
        for point in input:
            outputs = self.calculate(point)
            #euclid
            d = [np.subtract(x, point) for x in self.weights]
            d = [np.sum(np.abs(x)) for x in d]
            #append indexes to d
            d = [(d[i], i) for i in range(len(d))]
            #wta - only one best
            d = sorted(d, key=lambda x: x[0])
            network_errors += d[0][0]
            self.__adaptWeight(point, d[0][1])
        self.eta -= self.deta
        return network_errors

    def calculate(self, input):
        weighted_input = np.dot(self.weights, np.transpose(input))
        outputs = [self.activation(x) for x in weighted_input]
        return outputs

    def __adaptWeight(self, correct, neuron_index):
        current_neuron = self.weights[neuron_index]
        self.weights[neuron_index] = list(np.add(current_neuron, np.multiply(self.eta, np.subtract(correct, current_neuron))))
