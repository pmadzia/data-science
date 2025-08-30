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
    
    def frwd_prop(self, X):
        # Linear step: input -> hidden
        z1 = np.dot(X, self.W1) + self.b1  

        # Activation: ReLU
        A1 = np.maximum(0, z1)

        # Linear step: hidden -> output
        z2 = np.dot(A1, self.W2) + self.b2

        # Stable softmax (subtract max for numerical stability)
        z2_shifted = z2 - np.max(z2, axis=1, keepdims=True)
        exp_scores = np.exp(z2_shifted)
        A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return A1, A2


    def ce_loss(self, y_true, y_pred_proba):
        # Cross-entropy loss with epsilon for stability
        num_examples = y_true.shape[0]
        eps = 1e-15
        correct_log_probs = -np.log(y_pred_proba[range(num_examples), y_true] + eps)
        loss = np.sum(correct_log_probs) / num_examples
        return loss