import numpy as np
from TreeNode import TreeNode

class DecisionTree():
    """
    A decision tree classifier
    """

    def __init__(self, max_depth: int=2, min_samples: int=20, min_information_gain: float=1e-20, max_featrues: str="sqrt"):
        """
        The constructor for DecisionTree class.
        Args:
            max_depth: The maximum depth of the decision tree.
            min_samples: The minimum number of data samples required to split an internal node.
            min_information_gain: The minimum information gain required to split an internal node.
            max_features: The number of features to consider to build a tree. Default is "sqrt", the squared root of the total features.
                          User can also choose "log", the log of the total features. 
                          Otherwise all features are used.
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_information_gain = min_information_gain
        self.max_features = max_featrues
    
    def entropy(self, y: np.ndarray) -> float:
        """
        A function to calculate the entropy of the given numpy array.
        Args: 
            y: Input numpy array.
        Returns:
            entropy: Entropy of the given input values.
        """
        entropy = 0
        unique_values = np.unique(y)
        for value in unique_values:
            subset = y[y==value]
            p = len(subset) / len(y)
            entropy += -p*np.log2(p)
        return entropy
    
    def information_gain(self, parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        """
        A function to calculate the information gain from splitting the parent dataset into two datasets.
        Args:
            parent: Input parent dataset.
            left: The left child dataset after spliting on a feature.
            right: The right child dataset after spliting on a feature.
        Returns: 
            information_gain: The value of information gain of this split.
        """
        information_gain = 0
        parent_entropy = self.entropy(parent)
        left_weight = len(left) / len(parent)
        left_entropy = self.entropy(left)
        right_weight = len(right) / len(parent)
        right_entropy = self.entropy(right)

        information_gain = parent_entropy - left_weight * left_entropy - right_weight * right_entropy
        return information_gain
    
    def split_data(self, data: np.ndarray, feature: int, threshold: float) -> tuple:
        """
        A function to split data into two datasets based on the given feature and threshold.
        Args:
            data: The dataset to be split.
            feature: The feature index to be split on. 
            threshold: Threshold value to split the feature on.
        Returns:
            A tuple containing the left and right child datasets.
        """
        left = []
        right = []

        for row in data:
            if row[feature] < threshold:
                left.append(row)
            else:
                right.append(row)

        left = np.array(left)
        right = np.array(right)

        return (left, right)


    def find_best_split(self, feature: int, data: np.ndarray) -> dict:
        """
        A function to find the best split for the given feature in the given dataset.
        Args:
            feature: The feature index to be split on. 
            data: The dataset to be split.
        Returns:
            best_split: A dictionary with the best information gain and its threshold.
        """
        best_split = {"gain": -1, "threshold": None}
        feature_values = data[:, feature]
        thresholds = np.unique(feature_values)
        for threshold in thresholds[1:]:
            left, right = self.split_data(data, feature, threshold)
            if len(left) and len(right):
                y, y_left, y_right = data[:, -1], left[:, -1], right[:, -1]
                ig = self.information_gain(y, y_left, y_right)
                if ig > best_split["gain"]:
                    best_split["gain"] = ig
                    best_split["threshold"] = threshold
        return best_split
    
    def select_features(self, data: np.ndarray) -> list:
        """
        A function to select the features to use randomly.
        Args:
            data: The input dataset
        Returns:
            feature_idx: A list of the indices of the selected features.
        """
        feature_list = list(range(data.shape[1]-1))
        if self.max_features == "sqrt":
            feature_idx = np.random.choice(feature_list, size=int(np.sqrt(len(feature_list))))
        elif self.max_features == "log":
            feature_idx = np.random.choice(feature_list, size=int(np.log2(len(feature_list))))
        else:
            feature_idx = feature_list
        
        return feature_idx
    
    def select_feature_to_split(self, data: np.ndarray) -> dict:
        """
        A function to select the feature to split on. 
        Args:
            data: The dataset to be split.
        Returns: 
            best_feature: A dictionary with the best split feature index, threshold, and information gain.
        """
        best_feature = {"feature": None, "gain": -1, "threshold": None}
        features = self.select_features(data)
        ig_max = self.min_information_gain
        for feature in features:
            best_split = self.find_best_split(feature, data)
            if best_split["gain"] > ig_max:
                ig_max = best_split["gain"]
                best_feature["feature"] = feature
                best_feature["gain"] = best_split["gain"]
                best_feature["threshold"] = best_split["threshold"]

        return best_feature
    
    def caculate_leaf_value(self, y: np.ndarray):
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
    
    def build_tree(self, data: np.ndarray, current_depth: int=0) -> TreeNode:
        """
        This function recursively builds a decision tree from the given dataset.
        Args:
            data: The dataset to build the tree from.
            current_depth: The current depth of the tree.
        Returns:
            The root node of the built decision tree.
        """
        n_samples = data.shape[0]
        y = data[:, -1]
        if n_samples > self.min_samples and current_depth < self.max_depth:
            best_feature = self.select_feature_to_split(data)
            if best_feature["gain"] > self.min_information_gain:
                gain = best_feature["gain"]
                feature = best_feature["feature"]
                threshold = best_feature["threshold"]
                left, right = self.split_data(data, feature, threshold)
                left_node = self.build_tree(left, current_depth+1)
                right_node = self.build_tree(right, current_depth+1)
                return TreeNode(feature, threshold, left_node, right_node, gain)
            else:
                leaf_value = self.caculate_leaf_value(y)
                return TreeNode(value=leaf_value)
        
        leaf_value = self.caculate_leaf_value(y)
        return TreeNode(value=leaf_value)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        This function builds and fits the decision tree to the given feature matrix and target values.
        Args:
            X: The feature matrix.
            y: The target values.
        """
        data = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(data)

    def predict(self, X: np.ndarray):
        """
        This function predicts the class labels for each instance in the feature matrix X.
        Args:
            X: The feature matrix to make predictions for.
        Returns:
            A list of predicted class labels.
        """
        predictions = []
        for x in X:
            prediction = self.make_prediction(x, self.root)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return predictions
    
    def make_prediction(self, x: np.ndarray, node: TreeNode):
        """
        This function transverses the decision tree to predict the target value for the given feature vector.
        Args:
            x: The feature vector to predict the target value for.
            node: The current node being evaluated.
        Returns:
            The predicted target value for the given feature vector. 
        """
        if node.value != None:
            return node.value
        else:
            feature = x[node.feature]
            if feature < node.threshold:
                return self.make_prediction(x, node.left)
            else:
                return self.make_prediction(x, node.right)
    