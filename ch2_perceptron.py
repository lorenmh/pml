# -*- coding: utf-8 -*-

import numpy as np

''' Perceptron; heavily refactored from PML pg. 25 - 26
    >>> weights = a vector of weights
    >>> learning_rate = 'Eta'; the learning rate for the perceptron
    >>> iteratoins = the number of iterations to train the perceptron
'''
class Perceptron(object):
    def __init__(self, learning_rate=0.01, iterations=10):
        self.learning_rate = learning_rate
        self.iterations = iterations

    ''' Δw = η(yⁱ - ŷⁱ)xⁱ
        weight_delta = learning_rate(η) * (actual(y) - predicted(ŷ)) * input(x)
    '''
    def weight_delta(self, input, output_delta):
        return self.learning_rate * output_delta * input

    ''' inputs is a matrix; columns are features, rows are records

        outputs is a vector of classifications; 1 or -1

        xⁱ is the set of features corresponding to yⁱ

        weights is a one dimensional matrix of weights, with the first column
        being the negative of the threshold(-θ)

        if wᵀxⁱ >= 0 (with xⁱ⁰ being 1), then ŷⁱ is 1, else -1
    '''
    def train(self, inputs, outputs, num_features=None):
        # if num_features is not defined, then just get the number of features
        # from the first input element
        if num_features == None:
            num_features = len(inputs[0])

        self.weights = np.zeros(num_features + 1)

        c = 0
        for _ in range(self.iterations):
            # i is the current row count
            # xi is xⁱ; the set of features for row i
            for i, xi in enumerate(inputs):
                actual = outputs[i]
                predicted = self.predict(xi)

                delta = self.learning_rate * (actual - predicted)

                g = False
                if delta != 0:
                    print 'NE %d' % c
                    g = True

                self.weights[0] += delta
                self.weights[1:] += delta * xi

                if g:
                    print self.weights
                c += 1

    # if z >= 0, then activates to 1, else activates to -1
    def predict(self, x):
        z = np.dot(x, self.weights[1:]) + self.weights[0]
        return 1 if z >= 0 else -1
