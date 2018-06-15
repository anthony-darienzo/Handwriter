# Neural Network code to predict handwriting from MNIST database
# (C) 2018 Anthony D'Arienzo and Brevan Ellefsen

import numpy as np
import sympy as sy

from mnist import MNIST
mndata_training = MNIST('Training')
mndata_testing = MNIST('Testing')
training_images, training_labels = mndata_training.load_training()
test_images, test_labels = mndata_testing.load_testing()

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
NEURON_COUNT_LAYER1 = 30
NEURON_COUNT_LAYER2 = 15
INPUT_COUNT = IMAGE_WIDTH * IMAGE_HEIGHT
OUTPUT_COUNT = 10 # There are ten different digits (0,1,2,3,4,5,6,7,8,9)
NORMALIZATION_CONST = 255.0

LEARNING_RATE = 0.01
MOMENTUM_FACTOR = 0.9

def get_batches(iterable, batch_size):
    # Based on code from a StackOverflow answer by Carl F.
    # https://stackoverflow.com/questions/8290397/how-to-split-an-iterable-in-constant-size-chunks
    l = len(iterable)
    for chunk in range(0, l, batch_size):
        yield iterable[chunk:min(chunk + batch_size, l)]

def sigmoid(z):
    # The sigmoid activation function
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    # The derivative of the sigmoid activation function
    return np.multiply(sigmoid(z), np.ones(np.shape(z)) - sigmoid(z))

def mult(x, weights):
    # Take inputs and return the output activation for a matrix of weights
    return np.dot(x, weights)

def parse_label(numeral):
    training_vector = np.zeros(10)
    training_vector[numeral] = 255.0 / NORMALIZATION_CONST
    return training_vector

class neural_network:
    def __init__(self):
        '''
            X: Input Data vector
            O(n): The matrix that determines the input into the sigmoid function of layer n+1
            A(n): The matrix of activation values of the neurons in layer n
            W(n): The weights of the neurons in the n-th layer (Input is layer 0)
            V(n): The velocity vector for gradient descent
            Guess: The neural network's guess as to what numeral it "sees"
        '''
        self.W1 = np.asmatrix(np.random.rand(INPUT_COUNT,NEURON_COUNT_LAYER1) - 0.50)
        self.W2 = np.asmatrix(np.random.rand(NEURON_COUNT_LAYER1, NEURON_COUNT_LAYER2) - 0.50)
        self.W_out = np.asmatrix(np.random.rand(NEURON_COUNT_LAYER2, OUTPUT_COUNT) - 0.50)

        self.V1 = 0
        self.V2 = 0
        self.V_out = 0

    def propagate(self, input_data):
        self.X = np.asmatrix(input_data)
        self.O0 = mult(input_data, self.W1)
        self.A1 = sigmoid(self.O0)
        self.O1 = mult(self.A1, self.W2)
        self.A2 = sigmoid(self.O1)
        self.O_out = mult(self.A2, self.W_out)
        self.A_out = sigmoid(self.O_out)
        self.guess = [np.argmax(self.A_out[n]) for n in range(len(self.A_out))]

    def error(self, expected_data):
        self.residual = self.A_out - expected_data
        return self.residual

    def gradient(self):
        batch_size = self.residual.shape[0]
        self.delta_out = np.multiply(self.residual, sigmoid_prime(self.O_out))
        # We want to take the average deviation below, so we probably want to divide by the batch size
        #   ALSO: delta_out has dimension (batch_size, 10), A2 has dimension (batch_size,15)
        #       thus, the line below - through matrix multiplication - adds the deviations of every
        #       data value, leaving a matrix W_out_grad with dimension (15,10)
        self.W_out_grad = np.divide(np.dot(self.A2.T, self.delta_out), batch_size)
        self.delta_2 = np.multiply(np.dot(self.delta_out, self.W_out.T), sigmoid_prime(self.O1))
        self.W2_grad = np.dot(self.A1.T, self.delta_2)
        self.delta_1 = np.multiply(np.dot(self.delta_2, self.W2.T), sigmoid_prime(self.O0))
        self.W1_grad = np.dot(self.X.T, self.delta_1)

    def improve(self):

        self.V1 = (MOMENTUM_FACTOR * self.V1) + (LEARNING_RATE * self.W1_grad)
        self.V2 = (MOMENTUM_FACTOR * self.V2) + (LEARNING_RATE * self.W2_grad)
        self.V_out = (MOMENTUM_FACTOR * self.V_out) + (LEARNING_RATE * self.W_out_grad)

        self.W1 = self.W1 - self.V1
        self.W2 = self.W2 - self.V2
        self.W_out = self.W_out - self.V_out


image_train = np.divide(training_images, NORMALIZATION_CONST)
label_train = training_labels

trainingData = [[image_train[n],label_train[n]] for n in range(len(image_train))]

def naive_training(neuralNet, num_epochs):
    print("Training the network (Naive algorithm)")
    for k in range(num_epochs):
        print('Training epoch: ' + str(k+1))
        np.random.shuffle(trainingData)
        imageData = [trainingData[n][0] for n in range(len(trainingData))]
        labelData = np.asmatrix([parse_label(trainingData[n][1]) for n in range(len(trainingData))])
        for n in range(len(trainingData)):
            neuralNet.propagate(imageData[n])
            neuralNet.error(labelData[n])
            neuralNet.gradient()
            neuralNet.improve()

def BGD_training(neuralNet, num_epochs, batch_size):
    print("Training the network BGD Algorithm")
    for i in range(num_epochs):
        np.random.shuffle(trainingData)
        for batch in get_batches(trainingData, batch_size):
            imageData = [batch[n][0] for n in range(len(batch))]
            labelData = np.asmatrix([parse_label(batch[n][1]) for n in range(len(batch))])
            neuralNet.propagate(imageData)
            neuralNet.error(labelData)
            neuralNet.gradient()
            neuralNet.improve()

            
image_test = np.divide(test_images, NORMALIZATION_CONST)

def test(neuralNet):
    success_count = 0
    for n in range(10000):
        neuralNet.propagate(image_test[n])
        # We note, that the neural network outputs a guess vector containing its guesses for the entire
        # batch of data, thus, we want to take the first element of this array below (the test method
        #  has a batch size of 1 by design)
        if (neuralNet.guess[0] == test_labels[n]):
            success_count += 1
    print("Accuracy: " + str(success_count / 100.0) + "%") # Calculate % accuracy
    

potato = neural_network()

BGD_training(potato, 5, 10)
test(potato)

kiwi = neural_network()
naive_training(kiwi, 5)
test(kiwi)
