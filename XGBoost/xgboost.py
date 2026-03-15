import math
import numpy as np
from collections import defaultdict

class BoostedTree:

    def __init__(self, 
                 X: np.ndarray, 
                 gradients: np.ndarray, 
                 hessians: np.ndarray, 
                 params: dict, 
                 max_depth: int=2,
                 idxs: np.ndarray=None):
        """
        This class implements a single regression tree. The information gain of this tree is calculated by 
        gradients and hessians from the loss function.
        Args:
            X: The feature matrix for training.
            gradients: The first order derivatives of the loss function.
            hessians: The second order derivatives of the loss function.
            params: The dictionary of hyperparameters:
                min_child_weight: Minimum sum of hessian needed in a child node.
                reg_lambda: The L2 regularization term.
                reg_gamma: The minimum loss reduction to make a split.
            max_depth: The maximum depth of the tree.
            idxs: Indices of data samples that is used to build this tree.
        """
        self.X = X
        self.gradients = gradients
        self.hessians = hessians
        self.params = params
        self.min_child_weight = self.params['min_child_weight'] if self.params['min_child_weight'] else 1.0
        self.reg_lambda = self.params['reg_lambda'] if self.params['reg_lambda'] else 1.0
        self.reg_gamma = self.params['reg_gamma'] if self.params['reg_gamma'] else 0.0
        self.max_depth = max_depth
        self.ridxs = idxs if idxs is not None else np.arange(len(gradients))
        self.num_samples = len(self.ridxs)
        self.num_features = X.shape[1]
        self.weight = -self.gradients[self.ridxs].sum() / (self.hessians[self.ridxs].sum() + self.reg_lambda)
        self.best_score = 0.0
        self.split_idx = 0
        self.threshold = 0.0
        self.build_tree()

    def build_tree(self):
        """
        This function builds the tree recursively by finding the best splits.
        """
        if self.max_depth <= 0:
            return # Stop when reaching the max depth
        
        for fidx in range(self.num_features):
            self.find_best_split(fidx) # Try splitting on each feature to find the best split

        if self.is_leaf:
            return # If there is not valid split found, it is a leaf
        
        feature = self.X[self.ridxs, self.split_idx]
        left_idxs = np.nonzero(feature<=self.threshold)[0]
        right_idxs = np.nonzero(feature>self.threshold)[0]

        self.left = BoostedTree(self.X, self.gradients, self.hessians, self.params, self.max_depth-1, self.ridxs[left_idxs])
        self.right = BoostedTree(self.X, self.gradients, self.hessians, self.params, self.max_depth-1, self.ridxs[right_idxs])

    def find_best_split(self, fidx: int):
        """
        This function finds the best splits for a given feature.
        Args: 
            fidx: Index of the feature to evaluate for splitting.
        """
        feature = self.X[self.ridxs, fidx]
        gradients = self.gradients[self.ridxs]
        hessians = self.hessians[self.ridxs]

        sorted_idxs = np.argsort(feature)
        sorted_feature = feature[sorted_idxs]
        sorted_gradients = gradients[sorted_idxs]
        sorted_hessians = hessians[sorted_idxs]

        gradient_sum = sorted_gradients.sum()
        hessian_sum = sorted_hessians.sum()

        right_gradient_sum = gradient_sum
        right_hessian_sum = hessian_sum
        left_gradient_sum = 0.0
        left_hessian_sum = 0.0

        for idx in range(0, self.num_samples-1):
            candidate = sorted_feature[idx]
            neighbor = sorted_feature[idx+1]

            gradient = sorted_gradients[idx]
            hessian = sorted_hessians[idx]

            right_gradient_sum -= gradient
            right_hessian_sum -= hessian
            left_gradient_sum += gradient
            left_hessian_sum += hessian

            if left_hessian_sum <= self.min_child_weight or candidate == neighbor:
                continue

            if right_hessian_sum <= self.min_child_weight:
                break

            right_score = (right_gradient_sum**2) / (right_hessian_sum + self.reg_lambda)
            left_score = (left_gradient_sum**2) / (left_hessian_sum + self.reg_lambda)
            origin_score = (gradient_sum**2) / (hessian_sum+ self.reg_lambda)

            gain = 0.5 * (left_score + right_score - origin_score) - self.reg_gamma

            if gain > self.best_score:
                self.best_score = gain
                self.split_idx = fidx
                self.threshold = (candidate + neighbor) / 2

    def predict(self, X: np.ndarray):
        """
        This function makes predictions for a batch of data samples.
        Args:
            X: The feature matrix to make predictions on.
        Returns:
            A numpy array of predicted target values.
        """
        return np.array([self.predict_row(sample) for sample in X])
    
    def predict_row(self, sample: np.ndarray):
        """
        This function makes a prediction for a single data sample.
        Args:
            sample: A single data sample as an array of feature values.
        Returns:
            The predicted values of the given sample.
        """
        if self.is_leaf:
            return self.weight
        child = self.left if sample[self.split_idx] <= self.threshold else self.right
        return child.predict_row(sample)

    @property
    def is_leaf(self):
        """
        This function checks if the current node is a leaf node.
        Returns:
            True if this is a leaf node (no valid split found), False otherwise
        """
        return self.best_score == 0.0 # Leaf node if no gain found.

class XGBoostClassifier:
    def __init__(self, 
                 params: dict, 
                 seed: int=42):
        """
        This class implements XGBoost algorithm for binary classification problems.
        Args:
            params: The dictionary of hyperparameters:
                subsample: The fraction of data samples for training.
                n_estimators: The number of weak models (trees).
                learning_rate: The learning rate for training.
                max_depth: The maximum depth of a tree.
            seed: The random seed for reproducibility.
        """
        self.params = defaultdict(lambda: None, params)
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.n_estimators = params['n_estimators'] if params['n_estimators'] else 2
        self.learning_rate = params['learning_rate'] if params['learning_rate'] else 0.1
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 2
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def sigmoid(self, raw_predictions: np.ndarray):
        """
        The sigmoid function.
        """
        return 1 / (1 + np.exp(-raw_predictions))

    def compute_gradients(self, y: np.ndarray, prob: np.ndarray):
        """
        This function computes the gradients of binary cross-entropy.
        Args:
            y: The true target labels.
            prob: The predicted probabilities.
        Returns:
            The gradients.
        """
        return prob - y
    
    def compute_hessians(self, y: np.ndarray, prob: np.ndarray):
        """
        This function computes the hessians of binary cross-entropy.
        Args:
            y: The true target labels.
            prob: The predicted probabilities.
        Returns:
            The hessians.
        """
        return prob * (1 - prob)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This function fits the XGBoost classifier.
        Args:
            X: The feature matrix.
            y: The target labels.
        """
        raw_predictions = np.zeros(y.shape)
        probs = self.sigmoid(raw_predictions)
        for _ in range(self.n_estimators):
            gradients = self.compute_gradients(y, probs)
            hessians = self.compute_hessians(y, probs)

            if self.subsample == 1:
                idxs = None
            else:
                idxs = self.rng.choice(len(y), size=math.floor(self.subsample*len(y)), replace=False)
            tree = BoostedTree(X, gradients, hessians, self.params, self.max_depth, idxs)
            self.trees.append(tree)
            raw_predictions += self.learning_rate * tree.predict(X)
            probs = self.sigmoid(raw_predictions)

    def predict(self, X: np.ndarray):
        """
        This function combines the predictions of weak models (trees) and give the final predictions.
        Args:
            X: The feature matrix to make predictions on
        Returns:
            A numpy array of predicted target labels.
        """
        raw_predictions = np.zeros(shape=X.shape[0])
        for i in range(self.n_estimators):
            raw_predictions += self.learning_rate * self.trees[i].predict(X)
        probs = self.sigmoid(raw_predictions)
        return (probs > 0.5).astype(int)

class XGBoostRegressor:
    def __init__(self,
                 params: dict, 
                 seed: int=42):
        """
        This class implements XGBoost algorithm for regression problems.
        Args:
            params: The dictionary of hyperparameters:
                subsample: The fraction of data samples for training.
                n_estimators: The number of weak models (trees).
                learning_rate: The learning rate for training.
                max_depth: The maximum depth of a tree.
            seed: The random seed for reproducibility.        
        """
        self.params = defaultdict(lambda: None, params)
        self.subsample = self.params['subsample'] if self.params['subsample'] else 1.0
        self.n_estimators = params['n_estimators'] if params['n_estimators'] else 2
        self.learning_rate = params['learning_rate'] if params['learning_rate'] else 0.1
        self.max_depth = self.params['max_depth'] if self.params['max_depth'] else 2
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def compute_gradients(self, y: np.ndarray, pred: np.ndarray):
        """
        This function computes the gradients of mean squared error.
        Args:
            y: The true target values.
            pred: The predicted values.
        Returns:
            The gradients.
        """
        return pred - y

    def compute_hessians(self, y: np.ndarray, pred: np.ndarray):
        """
        This function computes the hessians of mean squared error. 
        Since the second-order derivative of this loss function is just 1, 
        This function just returns a numpy array with length of predictions.
        Args:
            y: The true target values.
            pred: The predicted values.
        Returns:
            The hessians.
        """
        return np.ones(shape=pred.shape)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This function fits the XGBoost regressor.
        Args:
            X: The feature matrix.
            y: The target values.
        """
        self.init_leaf = np.mean(y)
        predictions = np.zeros(y.shape) + self.init_leaf

        for _ in range(self.n_estimators):
            gradients = self.compute_gradients(y, predictions)
            hessians = self.compute_hessians(y, predictions)

            if self.subsample == 1:
                idxs = None
            else:
                idxs = self.rng.choice(len(y), size=math.floor(self.subsample*len(y)), replace=False)
            tree = BoostedTree(X, gradients, hessians, self.params, self.max_depth, idxs)
            self.trees.append(tree)
            predictions += self.learning_rate * tree.predict(X).reshape(-1, 1)

    def predict(self, X: np.ndarray):
        """
        This function combines the predictions of weak models (trees) and give the final predictions.
        Args:
            X: The feature matrix to make predictions on
        Returns:
            A numpy array of predicted target values.
        """
        predictions = np.zeros(shape=(X.shape[0], 1)) + self.init_leaf

        for i in range(self.n_estimators):
            predictions += self.learning_rate * self.trees[i].predict(X).reshape(-1, 1)
        
        return predictions