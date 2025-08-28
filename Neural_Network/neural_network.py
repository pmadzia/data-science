import numpy as np

# Implementation of Neural Network
class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size):
        # layer size
        self.input_size = input_size    # number of inputs
        self.hidden_size = hidden_size  # number of neurons in hidden layer
        self.output_size = output_size  # number of neurons in output layer

        # (input -> hidden)
        # weight matrix of shape (input_size, hidden_size)
        self.W1 = 0.01 * np.random.randn(self.input_size, self.hidden_size)
        # bias vector for hidden layer, shape (1, hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        # (hidden -> output)
        # weight matrix of shape (hidden_size, output_size)
        self.W2 = 0.01 * np.random.randn(self.hidden_size, self.output_size)
        # bias vector for output layer, shape (1, output_size)
        self.b2 = np.zeros((1, self.output_size))