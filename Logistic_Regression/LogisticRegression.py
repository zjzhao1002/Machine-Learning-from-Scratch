import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float=0.1, iterations: int=100, epsilon: float=1e-10):
        """
        The constructor for LogisticRegression class.
        Args:
            learning_rate: The learning rate of the model.
            iterations: The number of iterations of the model.
            epsilon: A small number to prevent issue from log(0).
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.epsilon = epsilon
        self.weights = None
        self.bias = 0

    def sigmoid(self, z: np.ndarray):
        """
        The sigmoid function for a given input.
        Args:
            z: Input numpy array.
        Returns:
            The results of sigmoid function.
        """
        return 1 / (1 + np.exp(-z))
    
    def cross_entropy(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        A function to compute the cross entropy loss.
        Args:
            y_true: The true labels for each data point.
            y_pred: The predicted probability for each data point.
        Returns:
            The value of cross entropy loss.
        """
        m = len(y_true)
        loss = - np.sum(y_true*np.log(y_pred + self.epsilon) + (1-y_true)*np.log(1-y_pred + self.epsilon))
        return loss / m
    
    def forward(self, X: np.ndarray):
        """
        Forward propagation of the model.
        Args:
            X: The feature matrix.
        Returns:
            The predictions of current iteration
        """
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return y_pred.reshape(-1, 1)
    
    def update_parameters(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        """
        Update the model parameters.
        Args: 
            X: The feature matrix.
            y: The true labels for each data point.
            y_pred: The predicted probability for each data point.
        """
        m = X.shape[0]
        dw = np.dot(X.T, (y_pred-y)) / m
        db = np.sum(y_pred-y) / m
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model on given input X and labels y.
        Args:
            X: The feature matrix.
            y: The true labels for each data point.            
        """
        _, n = X.shape
        self.weights = np.zeros((n,1))

        for i in range(self.iterations):
            y_pred = self.forward(X)

            self.update_parameters(X, y, y_pred)

            loss = self.cross_entropy(y, y_pred)
            if (i+1)%10 == 0:
                print(f"Running {i+1} iteration")
                print(f"loss = {loss}")
    
    def predict(self, X: np.ndarray):
        """
        Predicts the labels for given feature matrix X.
        Args:
            X: The feature matrix.
        Returns:
            The predicted labels.
        """
        predictions = self.forward(X)
        return (predictions >= 0.5).astype(int)
