import pandas as pd
import numpy as np
# from DecisionTree import DecisionTree
from RandomForest import RandomForest
from sklearn.preprocessing import OrdinalEncoder

def encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function detects the catagorical columns in a pandas dataframe, 
    and convert their values into numric values.
    Args:
        df: A pandas dataframe containing catagorical columns.
    Returns:
        df: A pandas dataframe with numeric columns only.
    """
    cat_columns = df.select_dtypes(include='object').columns
    
    encoder = OrdinalEncoder()
    df[cat_columns] = encoder.fit_transform(df[cat_columns].astype(str))
    
    return df

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

    data = pd.read_csv("500_Person_Gender_Height_Weight_Index.csv")
    data['obese'] = (data['Index'] >= 4).astype(int)
    data.drop('Index', axis=1, inplace=True)
    data = encoding(data)

    forest = RandomForest(num_trees=3, max_features="all", bootstrap_samples=None, max_depth=3)
    X = data.drop("obese", axis=1).values
    y = data['obese'].values.reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    forest.fit(X_train, y_train)
    train_preds = forest.predict(X_train)
    test_preds = forest.predict(X_test)

    print("The accuracy for the training data: ", accuracy(y_train, train_preds))
    print("The accuracy for the test data: ", accuracy(y_test, test_preds))