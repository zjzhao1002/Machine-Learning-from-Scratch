import numpy as np
from DecisionTree import DecisionTree

class AdaBoost():

    def __init__(self, n_estimators: int=2):
        '''
        The constructor for AdaBoost class.
        Args:
            n_estimators: The number of weak models in the model
        '''
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def calculate_alpha(self, sample_weight: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        '''
        A function to calculate the importance. 
        Args: 
            sample_weight: The weights of the current dataset
            y_true: The target values of the training dataset
            y_pred: The predictions of the current weak model
        Returns:
            The importance of the current weak model
        '''
        err = np.sum(sample_weight*(y_pred != y_true)) / np.sum(sample_weight)
        alpha = 0.5*np.log((1-err)/err)
        return alpha
    
    def update_weight(self, sample_weight: np.ndarray, alpha: float, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        '''
        A function to update the weights.
        Args:
            sample_weight: The weights of the current dataset
            alpha: the importance of the current model
            y_true: The target values of the training dataset
            y_pred: The predictions of the current weak model
        Returns:
            The weights for training in the next step
        '''
        sample_weight = sample_weight * np.exp(alpha * (np.not_equal(y_pred, y_true)).astype(int))
        sample_weight = sample_weight / np.sum(sample_weight)
        return sample_weight

    def fit(self, X: np.ndarray, y: np.ndarray):
        '''
        Train the Adaboost model
        Args: 
            X: The feature matrix
            y: The target values
        '''
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
        '''
        This function predicts the class labels for each instance in the feature matrix X.
        Args:
            X: The feature matrix to make predictions for.
        Returns:
            A list of predicted class labels.
        '''
        weighted_preds = np.zeros((X.shape[0], self.n_estimators))

        for i in range(self.n_estimators):
            predictions = self.models[i].predict(X)
            weighted_preds[:,i] += self.alphas[i] * predictions

        return np.argmax(weighted_preds, axis=1)


    


