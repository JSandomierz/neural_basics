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
        print("dEta:",self.deta)
        print(self.weights)

    def activation(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self, input):
        network_errors = 0
        for point in input:
            outputs = self.calculate(point)
            err = [np.subtract(x, point) for x in self.weights]
            err = [np.sum(np.abs(x)) for x in err]
            best_neuron = np.argmin(err)
            network_errors += err[best_neuron]
            # print('Best neuron:', best_neuron)
            # print("correct:", point)
            self.__adaptWeight(point, best_neuron)
        self.eta -= self.deta
        # print('next eta:',self.eta)
        # print("weights:", self.weights)
        #print('Network error:',network_err)
        return network_errors

    def calculate(self, input):
        weighted_input = np.dot(self.weights, np.transpose(input))
        # print("wo:",weighted_input)
        outputs = [self.activation(x) for x in weighted_input]
        # print("output",outputs)
        return outputs

    def __adaptWeight(self, correct, neuron_index):
        current_neuron = self.weights[neuron_index]
        # print("adapt:",self.weights)
        self.weights[neuron_index] = list(np.add(current_neuron, np.multiply(self.eta, np.subtract(correct, current_neuron))))
        # print("ADAPT:",self.weights)
