import pandas as pd
import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor

class GradientBoostingRegressor():
    def __init__(self, n_estimators: int=2, learning_rate: float=0.3, max_depth: int=2):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.initial_leaf = np.mean(y)
        predictions = np.zeros(y.shape) + self.initial_leaf
        
        for _ in range(self.n_estimators):
            residuals = y - predictions
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions += self.learning_rate * tree.predict(X).reshape(-1, 1)
            self.trees.append(tree)

    def predict(self, X: np.ndarray):
        predictions = np.zeros(shape=(len(X), 1)) + self.initial_leaf
        
        for i in range(self.n_estimators):
            predictions += self.learning_rate * self.trees[i].predict(X).reshape(-1, 1)
        
        return predictions