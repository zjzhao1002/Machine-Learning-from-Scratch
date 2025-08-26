import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self, n_features: int, tolerance: float=1.e-6):
        """
        The constructor for LinearRegression class
        Args: 
            n_features: The numbers of features in the input data
            tolerance: The tolerance for convergence (stopping criterion)
        """
        self.n_features = n_features
        self.tolerance = tolerance
        self.weights = np.random.randn(n_features).reshape(-1, 1) 
        self.bias = np.random.randn(1).reshape(-1, 1) 
        self.losses = []

    def forward(self, X: np.ndarray):
        """
        The forward pass of the linear regression model
        Args:
            X: The features matrix
        """
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        A function to compute the mean squared error
        Args: 
            y_true: The true target values
            y_pred: The predicted target values
        """
        m = y_true.shape[0]
        loss = np.sum(np.square(y_pred-y_true))/(2*m)
        return loss
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, X: np.ndarray):
        """
        The backward pass of the linear regression model
        Args:
            y_pred: The predicted target values
            y_true: The true target values
            X: The features matrix
        """
        m = y_true.shape[0]
        dw = np.dot(X.T, (y_pred-y_true)) / m
        db = np.sum(y_pred-y_true) / m
        return (dw, db)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int=10, eta: float=0.001):
        """
        This function fit the linear regression model
        Args:
            X: The features matrix
            y: The target values
            epochs: The number of iterations
            eta: The learning rate
        """
        if X.shape[1] != self.n_features:
            raise Exception("The number of features of your data is not consistent to the model.")
        
        for i in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)
            dw, db = self.backward(y_pred, y, X)
            self.weights -= eta*dw
            self.bias -= eta*db
            if (i+1)%100 == 0:
                print(f"Epoch: {i+1}/{epochs}, Loss: {loss}")

            if i>0 and abs(self.losses[-1] - self.losses[-2]) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break

    def predict(self, X: np.ndarray):
        """
        This function predict the target values with input X
        Args:
            X: The features matrix for prediction
        """
        return self.forward(X)
    
    def plotloss(self):
        """
        This function makes the plot of loss vs. epochs
        """
        epochs = np.arange(0, len(self.losses))
        plt.plot(epochs, self.losses)
        plt.xlabel('Number of Epochs')
        plt.ylabel('Loss')
        plt.title('Plot Loss')
        plt.show()
    