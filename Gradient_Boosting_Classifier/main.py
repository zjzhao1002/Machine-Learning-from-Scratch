import pandas as pd
import numpy as np
from GradientBoostingClassifier import GradientBoostingClassifier

def train_test_split(X: np.ndarray, y: np.ndarray, random_state: int=41, test_size: float=0.2) -> tuple:
    """
    This function splits data into training and test sets.
    Args:
        X: Features array.
        y: Target array.
        random_state: Seed for the random number generator.
        test_size: Proportion of samples to include in the test set. 
    Returns:
        A tuple that contains training and test datasets.
    """
    n_samples = X.shape[0]
    np.random.seed(random_state)

    shuffled_indices = np.random.permutation(np.arange(n_samples))

    test_size = int(n_samples * test_size)

    test_indices = shuffled_indices[:test_size]
    train_indices = shuffled_indices[test_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return (X_train, X_test, y_train, y_test)

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    This function calculate the accuracy of a classification model.
    Args:
        y_true: The true labels for each data point.
        y_pred: The predicted labels for each data point.
    Returns:
        The accuracy of the model.
    """
    y_true = y_true.flatten()
    total_samples = len(y_true)
    return (np.sum(y_true == y_pred)) / total_samples

if __name__ == "__main__":

    df = pd.read_csv("Iris.csv")

    X = df.drop(["Id", "Species"], axis=1).values
    y = df["Species"].replace(["Iris-setosa", "Iris-versicolor", "Iris-virginica"], [0, 1, 2])
    y = y.values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = GradientBoostingClassifier(n_estimators=3, learning_rate=0.1, max_depth=2)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    print("The accuracy for the training data: ", accuracy(y_train, predictions))
    predictions = model.predict(X_test)
    print("The accuracy for the test data: ", accuracy(y_test, predictions))