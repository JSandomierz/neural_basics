import numpy as np
import random
import math
from functools import reduce

class WTM:
    weights = []
    biases = []
    eta = 0.4
    deta = 0.98
    affected_distance = 0.3

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
            d = sorted(d, key=lambda x: x[0])
            #calculate distances from winning neuron to others
            best_neuron = d[0][1]
            distance_winning_to_others = [np.subtract(x, self.weights[best_neuron]) for x in self.weights]
            distance_winning_to_others = [np.sum(np.abs(x)) for x in distance_winning_to_others]
            #print("Winning neuron {}".format(best_neuron))
            #print("Point {}".format(point))
            #print("Winning neuron weights {}".format(self.weights[best_neuron]))
            #print("Neurons {}".format(self.weights))
            #print("Distance to others {}".format(distance_winning_to_others))
            #wtm
            affected_neurons = 0
            network_errors += d[0][0]
            for dist, i in d:
                if distance_winning_to_others[i] <= self.affected_distance:
                    affected_neurons += 1
                    #print("Updating neuron {}, dist: {}".format(i, dist))
                    self.__adaptWeight(point, i)
        self.eta -= self.deta
        #print("Affected neurons: ",affected_neurons," distance: ",self.affected_distance)
        self.affected_distance *= 0.98
        return network_errors

    def calculate(self, input):
        weighted_input = np.dot(self.weights, np.transpose(input))
        outputs = [self.activation(x) for x in weighted_input]
        return outputs

    def __adaptWeight(self, correct, neuron_index):
        current_neuron = self.weights[neuron_index]
        self.weights[neuron_index] = list(np.add(current_neuron, np.multiply(self.eta, np.subtract(correct, current_neuron))))
