import pandas as pd 
import numpy as np
from GradientBoostingRegressor import GradientBoostingRegressor

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

def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true-y_pred)**2)

if __name__ == "__main__":
    df = pd.read_csv("Student_Performance.csv")

    df = df.replace(['No', 'Yes'], [0, 1])

    X = df.drop("Performance Index", axis=1).values
    y = df["Performance Index"].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = GradientBoostingRegressor(n_estimators=6, max_depth=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    mse = compute_mse(y_train, predictions)
    print("MSE = ", mse)
    predictions = model.predict(X_test)
    mse = compute_mse(y_test, predictions)
    print("MSE = ", mse)
    print(y_test[:10])
    print(predictions[:10])