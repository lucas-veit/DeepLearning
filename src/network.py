# -*- coding: utf-8 -*-
import src.helpers as helpers
from enum import Enum
import numpy as np
import random

# Define the Enum for seed options
class SeedOption(Enum):
    NO_SEED = 1
    BGD_SEED = 2
    SGD_SEED = 3

class Network(object):

    def __init__(self, layers, seed_option=SeedOption.NO_SEED):
        # Batch Gradient Descent (BGD) is deterministic because it uses the entire training dataset in each update, regardless of the order.
        # Stochastic Gradient Descent (SGD) is not supposed to be deterministic because it involves shuffling the data and creating mini-batches,
        # which introduces randomness into the optimization process. This randomness helps prevent the model from getting stuck in local minima.
        # Initializing with a fixed seed is helpful for debugging and reproducibility in BGD, as it ensures that weights and biases are always initialized with the same random values.
        # However, SGD should remain non-deterministic by design. If you use np.random.shuffle instead of random.shuffle for shuffling the data, SGD would become deterministic,
        # but this goes against the intended behavior of SGD. If want to use the full power of SGD, then no seed to fix here.
        if seed_option in [SeedOption.BGD_SEED, SeedOption.SGD_SEED]:
            np.random.seed(42)  # Fixed seed for weights and biases random initialization
            if seed_option == SeedOption.SGD_SEED:
                random.seed(42)  # Fixed seed for random.shuffle used for SGD

        # List:
        # Containing the number of neurons in the respective layers
        # of the network, thus the input layer + hidden layers + output layer
        self.__layers = layers

        # Biases 2D matrix:
        # column is layer number, row is the neuron number on this layer
        self.__biases = [np.random.randn(y, 1) for y in layers[1:]]

        # Weights 3D matrix:
        # one table per layer, each table is a 2D matrix with column the src neuron, and row the dst neuron
        self.__weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def run(self, x):
        """
        Return the output of the network given the inputs x
        """
        activations, _ = self.__feedforward(x)
        return activations[-1]

    def training(self, training_data, epochs, eta, mini_batch_size=None, test_data=None):
        """
        Train the network using mini-batch stochastic gradient descent.
        The ``training_data`` is a list of tuplesc``(x, y)`` representing
        the training inputs and the desiredcoutputs.
        If ``mini_batch_size`` is provided then the stochastic gradient descent is used,
        otherwise it will be the batch gradient descent.
        If ``test_data`` is provided then the network will evaluate the test data
        after each epoch, and partial progress printed out.
        """
        for epoch_i in range(epochs):
            if mini_batch_size:
                # Shuffle the training data at each epoch to not having same mini_batches everytimes
                random.shuffle(training_data)
                # Divided the training_data into several mini_batch of size mini_batch_size
                mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, len(training_data), mini_batch_size)]
                for mini_batch in mini_batches:
                    # Update the network wieghts and biases based on a small set of data
                    # So update is done several times during an epoch
                    self.__update_parameters(mini_batch, eta)
            else :
                # Update the network weights and biases based on the whole training_data
                # So update is done once at each epoch
                self.__update_parameters(training_data, eta)
            
            # Output the log if a test_data is provided
            if test_data:
                print("Epoch {}: {} / {}".format(epoch_i, self.evaluate(test_data), len(test_data)))
            else:
                print("Epoch {} complete".format(epoch_i))

    def __update_parameters(self, training_dataset, eta):
        """
        Update the network's weights and biases by applying gradient descent using backpropagation.
        The ``training_dataset`` is a list of tuples ``(x, y)``, and ``eta`` is the learning rate.
        """
        # Initialization of the partial derivatives of weights and biases of the network
        nabla_b = [np.zeros(b.shape) for b in self.__biases]
        nabla_w = [np.zeros(w.shape) for w in self.__weights]

        # For each data, compute the partial derivative of weights and biases
        for x, y in training_dataset:
            # Get the partial derivative of weight and bias for the data using the backpropagation algorithm
            delta_nabla_b, delta_nabla_w = self.__backpropagation(x, y)
            # Sum of the partial derivative of each data stored in nabla
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # When all partial derivatives are computed, update the weights and biases by removing the mean
        # of all the partial derivative computed and applying a learning rate eta for the speed of convergence
        # The param nabla are the sum of the partial derivative of all data, so juste need to divide it by the
        # number of data to get the mean
        self.__weights = [w-(eta/len(training_dataset))*nw
                        for w, nw in zip(self.__weights, nabla_w)]
        self.__biases = [b-(eta/len(training_dataset))*nb
                       for b, nb in zip(self.__biases, nabla_b)]

    def __backpropagation(self, x, y):
        """
        Back propagation algorithme applied for a data with x as input and y as output.
        Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_i.
        ``nabla_b`` and ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.__biases`` and ``self.__weights``.
        """
        # Initialization of the partial derivatives of weights and biases of the network for the data
        nabla_b = [np.zeros(b.shape) for b in self.__biases]
        nabla_w = [np.zeros(w.shape) for w in self.__weights]

        # feedforward
        activations, zs = self.__feedforward(x)
        
        # backward
        # BP1
        delta = helpers.cost_derivative(activations[-1], y) * helpers.sigmoid_prime(zs[-1])
        # BP3 and BP4 for output layer
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, len(self.__layers)):
            z = zs[-l]
            sp = helpers.sigmoid_prime(z)
            # BP2
            delta = np.dot(self.__weights[-l+1].transpose(), delta) * sp
            # BP3 and BP4 for hidden layers
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def __feedforward(self, x):
        """
        For a given input x, compute the activations and errors z of the network
        WARNING : x should be a numpy array type
        """
        if len(x) != self.__layers[0]:
            raise Exception(f"Input size ({len(x)}) is not equal to the number of neurons in input layer ({self.__layers[0]})")

        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.__biases, self.__weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = helpers.sigmoid(z)
            activations.append(activation)
        return activations, zs
