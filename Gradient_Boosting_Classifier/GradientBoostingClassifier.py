import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder

class GradientBoostingClassifier():
    def __init__(self, n_estimators: int=2, learning_rate: float=0.2, max_depth: int=2):
        """
        The constructor for GradientBoostingClassifier class.
        Args:
            n_estimarots: The number of weak models in the model.
            learning_rate: The learning rate.
            max_depth: The maximum depth of the decision tree.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    def one_hot_encode(self, y: np.ndarray):
        """
        The function for one hot encoding.
        Args:
            y: The numpy array for encoding.
        Returns:
            y_ohe: The numpy array after one hot encoding.
        """
        ohe = OneHotEncoder()
        y_ohe = ohe.fit_transform(y).toarray()
        return y_ohe
    
    def softmax(self, raw_predictions: np.ndarray):
        """
        The softmax function.
        Args:
            raw_predictions: The numpy array to calculate the softmax function.
        Returns:
            The probability distribution converted from input array.
        """
        numerator = np.exp(raw_predictions)
        denominator = np.sum(np.exp(raw_predictions), axis=1).reshape(-1, 1)
        return numerator / denominator
    
    def negative_gradient(self, y_ohe: np.ndarray, probabilities: np.ndarray):
        """
        This function computes the negative gradient.
        Args:
            y_ohe: The true target labels after one hot encoded.
            probabilities: The predicted probabilities.
        Returns:
            The negative gradient.
        """
        return y_ohe - probabilities
    
    def hessians(self, probabilities: np.ndarray):
        """
        This function computes the second derivative of loss function.
        Args: 
            probabilities: The array of the predicted probabilities.
        """
        return probabilities*(1-probabilities)

    def update_leaf_nodes(self, tree: DecisionTreeRegressor, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        """
        This function updates the leaf nodes of the given decision tree.
        Args:
            tree: The decision tree to be updated.
            X: The data to be determined where they belong to which leaf nodes.
            gradients: The negative gradients of the loss function.
            hessians: The second derivatives of loss function.
        """
        leaf_nodes = np.nonzero(tree.tree_.children_left == -1)[0]
        leaf_node_for_each_example = tree.apply(X)
        for leaf in leaf_nodes:
            samples_in_this_leaf = np.where(leaf_node_for_each_example == leaf)[0]
            gradients_in_leaf = gradients.take(samples_in_this_leaf, axis=0)
            hessians_in_leaf = hessians.take(samples_in_this_leaf, axis=0)
            val = np.sum(gradients_in_leaf) / np.sum(hessians_in_leaf)
            tree.tree_.value[leaf, 0, 0] = val

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This function fits the gradient boosting model.
        Args: 
            X: The feature matrix.
            y: The target labels.
        """
        self.n_classes = len(np.unique(y))
        self.trees = {k: [] for k in range(self.n_classes)}
        y_ohe = self.one_hot_encode(y)
        raw_predictions = np.zeros(y_ohe.shape)

        for _ in range(self.n_estimators):
            probabilities = self.softmax(raw_predictions)

            for k in range(self.n_classes):
                gradients = self.negative_gradient(y_ohe[:,k], probabilities[:, k])
                hessians = self.hessians(probabilities[:, k])

                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, gradients)
                self.update_leaf_nodes(tree, X, gradients, hessians)
                raw_predictions[:, k] += self.learning_rate * tree.predict(X)
                probabilities = self.softmax(raw_predictions)
                self.trees[k].append(tree)

    def predict(self, X: np.ndarray):
        """
        This function combines the predictions of weak models and give the final predictions.
        Args: 
            X: The feature matrix to make predictions for.
        Returns:
            A numpy array of predicted target labels.
        """
        raw_predictions = np.zeros(shape=(X.shape[0], self.n_classes))
        for i in range(self.n_estimators):
            for k in range(self.n_classes):
                raw_predictions[:, k] += self.learning_rate * self.trees[k][i].predict(X)

        probabilities = self.softmax(raw_predictions)

        return np.argmax(probabilities, axis=1)