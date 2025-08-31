import numpy as np

class Neural_Network:
    '''A simple fully-connected neural network with one hidden layer,
    implemented using NumPy.

    This implementation supports:
    - ReLU activation for the hidden layer
    - Softmax activation for the output layer
    - Cross-entropy loss
    - L2 regularization
    - Gradient descent optimization
    '''
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initializes the neural network parameters.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of neurons in output layer.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weights and biases
        self.W1 = 0.01 * np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))

        self.W2 = 0.01 * np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
    
    def frwd_prop(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Performs forward propagation.

        Args:
            x (np.ndarray): Input data of shape (num_samples, input_size).

        Returns:
            tuple: (z1, A1, A2)
                - z1 (np.ndarray): Pre-activation values of hidden layer.
                - A1 (np.ndarray): Hidden layer activations after ReLU.
                - A2 (np.ndarray): Output probabilities after softmax.
        """
        # Input -> Hidden
        z1 = np.dot(x, self.W1) + self.b1  
        # Activation: ReLU
        A1 = np.maximum(0, z1)

        # Hidden -> Output
        z2 = np.dot(A1, self.W2) + self.b2
        # Stable softmax
        z2_shifted = z2 - np.max(z2, axis=1, keepdims=True)
        exp_scores = np.exp(z2_shifted)
        A2 = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return z1, A1, A2


    def ce_loss(self, y: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Computes cross-entropy loss.

        Args:
            y (np.ndarray): True labels (integers), shape (num_samples,).
            y_pred_proba (np.ndarray): Predicted probabilities,
                shape (num_samples, output_size).

        Returns:
            float: Average cross-entropy loss.
        """
        num_examples = y.shape[0]
        eps = 1e-15
        correct_log_probs = -np.log(y_pred_proba[range(num_examples), y] + eps)
        loss = np.sum(correct_log_probs) / num_examples
        return loss
    
    def backward_prop(self, x: np.ndarray, y: np.ndarray, 
                      z1: np.ndarray, A1: np.ndarray,
                      A2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Performs backward propagation to compute gradients.

        Args:
            x (np.ndarray): Input data, shape (num_samples, input_size).
            y (np.ndarray): True labels, shape (num_samples,).
            z1 (np.ndarray): Pre-activation values of hidden layer.
            A1 (np.ndarray): Hidden layer activations.
            A2 (np.ndarray): Predicted probabilities from softmax.

        Returns:
            tuple: (dW1, db1, dW2, db2)
                - dW1 (np.ndarray): Gradient of W1.
                - db1 (np.ndarray): Gradient of b1.
                - dW2 (np.ndarray): Gradient of W2.
                - db2 (np.ndarray): Gradient of b2.
        """
        num_examples = y.shape[0]

        # dL/dz2 = P - Y
        dZ2 = A2.copy()
        dZ2[range(num_examples), y] -= 1
        dZ2 /= num_examples

        # Gradients for W2, b2
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Backprop into hidden layer
        dA1 = np.dot(dZ2, self.W2.T)
        # ReLU gradient (zero out where z1 <= 0)
        dA1[z1 <= 0] = 0

        # Gradients for W1, b1
        dW1 = np.dot(x.T, dA1)
        db1 = np.sum(dA1, axis=0, keepdims=True)

        return dW1, db1, dW2, db2
    
    def fit(self, x: np.ndarray, y: np.ndarray, 
            reg: float, epochs: int, eta: float):
        """Trains the neural network using gradient descent.

        Args:
            x (np.ndarray): Training data, shape (num_samples, input_size).
            y (np.ndarray): True labels, shape (num_samples,).
            reg (float): L2 regularization strength.
            epochs (int): Number of iterations.
            eta (float): Learning rate.
        """
        for i in range(epochs):
            # Forward propagation
            z1, A1, A2 = self.frwd_prop(x)

            # Compute loss
            loss = self.ce_loss(y, A2)
            reg_loss = 0.5 * reg * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
            total_loss = loss + reg_loss

            # Print every 1000 steps
            if i % 1000 == 0:
                print(f"iteration {i}: loss {total_loss:.6f}")

            # Backpropagation
            dW1, db1, dW2, db2 = self.backward_prop(x, y, z1, A1, A2)

            # Add regularization gradients
            dW1 += reg * self.W1
            dW2 += reg * self.W2

            # Parameter update
            self.W1 -= eta * dW1
            self.b1 -= eta * db1
            self.W2 -= eta * dW2
            self.b2 -= eta * db2

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts class labels for input data.

        Args:
            x (np.ndarray): Input data, shape (num_samples, input_size).

        Returns:
            np.ndarray: Predicted class labels, shape (num_samples,).
        """
        _, _, A2 = self.frwd_prop(x)
        y_pred = np.argmax(A2, axis=1)

        return y_pred
