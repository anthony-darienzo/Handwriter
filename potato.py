import numpy as np
import sympy as sy
from sympy import *

from mnist import MNIST
mndata_training = MNIST('Training')
training_images, training_labels = mndata_training.load_training()

sy.init_printing(wrap_line=False)

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NEURON_COUNT = 10
INPUT_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT_COUNT = 10 # There are ten different digits (0,1,2,3,4,5,6,7,8,9)

def sigmoid(z):
    # The sigmoid activation function
    return 1 / (1+ np.exp(-z))

def sigmoid_prime(z):
    # The derivative of the sigmoid activation function
    return np.multiply(sigmoid(z), np.ones(np.shape(z)) - sigmoid(z))

def mult(x, weights):
    # Take inputs and return the output activation for a matrix of weights
    return np.dot(x, weights)

def parse_label(numeral):
    training_vector = np.zeros(10)
    training_vector[numeral] = 1
    return training_vector



class neural_network:
    def __init__(self):
        self.weights_hidden_layer = np.asmatrix(np.random.rand(INPUT_COUNT,NEURON_COUNT))
        self.weights_outer_layer = np.asmatrix(np.random.rand(NEURON_COUNT, OUTPUT_COUNT))

    def propagate(self, input_data):
        self.X = np.asmatrix(input_data)
        self.input_hidden_layer = mult(input_data, self.weights_hidden_layer)
        self.activation_hidden_layer = sigmoid(self.input_hidden_layer)
        self.input_output_layer = mult(self.activation_hidden_layer, self.weights_outer_layer)
        self.activation_output_layer = sigmoid(self.input_output_layer)
        self.guess = np.argmax(self.activation_output_layer)

    def calc_error(self, expected_data):
        self.residual = self.activation_output_layer - expected_data
        return self.residual

    def gradient(self):
        self.delta_2 = np.multiply(self.residual, sigmoid_prime(self.input_output_layer))
        self.weights_outer_layer_grad = np.dot(self.activation_hidden_layer.T, self.delta_2)
        self.delta_1 = np.multiply(np.dot(self.delta_2, self.weights_outer_layer.T), sigmoid_prime(self.input_hidden_layer))
        self.weights_hidden_layer_grad = np.dot(self.X.T, self.delta_1)

    def improve(self):
        self.weights_hidden_layer = self.weights_hidden_layer - self.weights_hidden_layer_grad
        self.weights_outer_layer = self.weights_outer_layer - self.weights_outer_layer_grad

neural_net = neural_network()

image = training_images
label = training_labels

neural_net.propagate(image[0])
before = neural_net.activation_output_layer

for n in range(10000):
    neural_net.propagate(image[n])
    neural_net.calc_error(parse_label(label[n]))
    neural_net.gradient()
    # Output the network's guess, alongside the true numeral
    print(str(neural_net.guess) + "<" + str(label[n]) + ">")
    # Update the weights for each neuron
    neural_net.improve()