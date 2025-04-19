import numpy as np

def sigmoid(z):
    """
    The sigmoid function used as activation function of each neuron
    """
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))

def cost_derivative(output_activations, y):
    """
    Return the vector of partial derivatives (partial C_x / partial a) for the output activations.
    """
    return (output_activations-y)
