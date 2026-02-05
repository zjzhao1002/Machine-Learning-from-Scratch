class TreeNode():
    """
    A class representing a node in a decision tree.
    """

    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        """
        The constructor for the TreeNode class.

        Args:
            feature: The feature used for splitting at this node.
            threshold: The threshold used for splitting at this node.
            left: The left child node.
            right: The right child node.
            gain: The information gain of the split.
            value: If this node is a leaf node, this attribute represents the predicted value for the target variable.
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.gain = gain
        self.value = value