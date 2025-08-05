import numpy as np
from DecisionTree import DecisionTree

class RandomForest():

    def __init__(self, 
                 num_trees: int=2, 
                 max_features: str="sqrt", 
                 bootstrap_samples: int=None,
                 max_depth: int=2, 
                 min_samples: int=20,
                 min_infomation_gain: float=1e-20):
        """
        The constructor for RandomForest class.
        Args:
            num_trees: The number of trees.
            max_features: The number of features to consider to build a tree. Default is "sqrt", the squared root of the total features.
                          User can also choose "log", the log of the total features. 
                          Otherwise all features are used.
            bootstrap_samples: The size of the bootstrap samples. If it is None, the size of the dataset is used.
            max_depth: The maximum depth of the decision tree.
            min_samples: The minimum number of data samples required to split an internal node.
            min_information_gain: The minimum information gain required to split an internal node.
        """
        self.num_trees = num_trees
        self.max_features = max_features
        self.bootstrap_samples = bootstrap_samples
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_information_gain = min_infomation_gain
        self.trees = []

    def create_bootstrap_samples(self, data: np.ndarray) -> np.ndarray:
        """
        This function create the bootstraping dataset by sampling from it with replacement.
        Args:
            data: The dataset to bootstrap
        Returns:
            data_sample: The bootstrapped data sample.
        """
        n_samples = data.shape[0]
        np.random.seed(10)
        if self.bootstrap_samples == None:
            indices = np.random.choice(n_samples, n_samples, replace=True)
        else:
            indices = np.random.choice(n_samples, self.bootstrap_samples, replace=True)

        data_sample = data[indices]
        return data_sample
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Build a random forest classifier from the training dataset (X, y).
        Args:
            X: The feature matrix.
            y: The target values.
        """
        self.trees = []
        data = np.concatenate((X, y), axis=1)
        for _ in range(self.num_trees):
            tree = DecisionTree(self.max_depth, self.min_samples, self.min_information_gain, self.max_features)
            data_sample = self.create_bootstrap_samples(data)
            X_sample, y_sample = data_sample[:, :-1], data_sample[:, -1]
            tree.fit(X_sample, y_sample.reshape(-1, 1))
            self.trees.append(tree)

    def most_common_label(self, y: np.ndarray):
        """
        A function to calculate the most occuring value in the given list of target values.
        Args:
            y: The list of target values.
        Returns:
            The most occuring value in this list.
        """
        y = list(y)
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    def predict(self, X: np.ndarray):
        """
        This function predicts the class labels for each instance in the feature matrix X.
        Args:
            X: The feature matrix to make predictions for.
        Returns:
            A list of predicted class labels.
        """
        predictions = np.array([tree.predict(X) for tree in self.trees])
        preds = np.swapaxes(predictions, 0, 1)
        majority_predictions = np.array([self.most_common_label(pred) for pred in preds])
        return majority_predictions
    
