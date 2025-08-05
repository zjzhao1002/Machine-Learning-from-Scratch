import numpy as np
from DecisionTree import DecisionTree

class AdaBoost():

    def __init__(self, n_estimators: int=2):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def calculate_alpha(self, sample_weight: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        err = np.sum(sample_weight*(y_pred != y_true)) / np.sum(sample_weight)
        alpha = 0.5*np.log((1-err)/err)
        return alpha
    
    def update_weight(self, sample_weight: np.ndarray, alpha: float, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        sample_weight = sample_weight * np.exp(alpha * (np.not_equal(y_pred, y_true)).astype(int))
        sample_weight = sample_weight / np.sum(sample_weight)
        return sample_weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples) / n_samples
        for _ in range(self.n_estimators):
            model = DecisionTree(max_depth=1)
            model.fit(X, y, sample_weight)
            predictions = model.predict(X)

            alpha = self.calculate_alpha(sample_weight, y, predictions)
            sample_weight = self.update_weight(sample_weight, alpha, y, predictions)
            
            self.models.append(model)
            self.alphas.append(alpha)

    def predict(self, X: np.ndarray):
        weighted_preds = np.zeros((X.shape[0], self.n_estimators))

        for i in range(self.n_estimators):
            predictions = self.models[i].predict(X)
            weighted_preds[:,i] += self.alphas[i] * predictions

        return np.argmax(weighted_preds, axis=1)


    


