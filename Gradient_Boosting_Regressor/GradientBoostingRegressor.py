import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoostingRegressor():
    def __init__(self, n_estimators: int=2, learning_rate: float=0.3, max_depth: int=2):
        """
        The constructor for GradientBoostingRegressor class.
        Args:
            n_estimarots: The number of weak models in the model.
            learning_rate: The learning rate.
            max_depth: The maximum depth of the decision tree.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This function builds and fits the model to the given feature matrix and target values.
        Args:
            X: The feature matrix.
            y: The target values.
        """
        self.initial_leaf = np.mean(y)
        predictions = np.zeros(y.shape) + self.initial_leaf
        
        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions += self.learning_rate * tree.predict(X).reshape(-1, 1)
            self.trees.append(tree)

    def predict(self, X: np.ndarray):
        """
        This function combines the predictions of weak models and give the final predictions.
        Args: 
            X: The feature matrix to make predictions for.
        Returns:
            A numpy array of predicted target values.
        """
        predictions = np.zeros(shape=(len(X), 1)) + self.initial_leaf
        
        for i in range(self.n_estimators):
            predictions += self.learning_rate * self.trees[i].predict(X).reshape(-1, 1)
        
        return predictions
