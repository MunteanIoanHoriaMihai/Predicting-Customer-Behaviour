"""The 'train_model' module includes functions to help train a chosen model with the corresponding data."""
from typing import Tuple, Any

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import os
from pandas import read_csv, DataFrame


def get_dataset(data_subdirectory: str) -> DataFrame:
    """
    Extract the data from a csv file found inside a subdirectory of the data directory.

    :param str data_subdirectory: The name of the subdirectory inside the data directory where the dataset is found
    :return: Returns a variable containing the dataset
    """
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', data_subdirectory, 'Mall_Customers.csv')
    names = ['CustomerID', 'Genre', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    retrieved_dataset = read_csv(path, names=names, skiprows=1)
    return retrieved_dataset


def make_train_test_datasets(retrieved_dataset, train_size: float = 0.8, test_size: float = 0.2,
                             random_state: int = 10) -> Tuple[Any, Any, Any, Any]:
    """
    Split the original dataset into a train dataset and a test dataset and separate the ground truths.

    :param retrieved_dataset: The parameter holding the original dataset
    :param float train_size: The size of the train dataset relative to the original dataset
    :param float test_size: The size of the test dataset relative to the original dataset
    :param float random_state: Parameter that controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
    :return: Returns the train and test datasets and their ground truths
    """
    df_train, df_test = train_test_split(retrieved_dataset,
                                         train_size=train_size,
                                         test_size=test_size,
                                         random_state=random_state)

    X_train = df_train.loc[:, ['Age', 'Annual Income (k$)']]
    y_train = df_train.loc[:, ['Spending Score (1-100)']]

    X_test = df_test.loc[:, ['Age', 'Annual Income (k$)']]
    y_test = df_test.loc[:, ['Spending Score (1-100)']]

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    retrieved_dataset = get_dataset('external')

    X_train, y_train, X_test, y_test = make_train_test_datasets(retrieved_dataset, train_size=0.8,
                                                                test_size=0.2, random_state=10)

    linear_regression = LinearRegression()
    linear_regression.fit(X_train, y_train)
    y_predicted = linear_regression.predict(X_test)

    print("\n", type(y_predicted))
    print("\n", type(y_test))
    y_test = y_test.to_numpy()

    print(mean_absolute_error(y_test, y_predicted))
    print(mean_squared_error(y_test, y_predicted))
