# ====== Base class imports ======

import time
from typing import Literal

# ====== Scikit-learn imports ======

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def scaler_fn(X, scaling: Literal['None', 'StandardScaler', 'MinMaxScaler']="None"):
    """
    This function applies scaling to the input data frame X based on the specified scaling method.
    It supports three scaling methods: 'None', 'StandardScaler', and 'MinMaxScaler'.
    If 'None' is specified, no scaling is applied and the original data is returned.
    If 'StandardScaler' is specified, the data is standardized (mean=0, variance=1).
    If 'MinMaxScaler' is specified, the data is scaled to a range between 0 and 1.

    Args:
        X (array-like): The input data to be scaled.
        scaling (str): The scaling method to be applied. Options are 'None', 'StandardScaler', 'MinMaxScaler'.

    Returns:
        X_scaled (array-like): The scaled data.
    """
    if scaling == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    if scaling == 'StandardScaler':
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    if scaling == 'None':
        X_scaled = X
    return X_scaled

def feature_encoding(feature1, sparse_output=False, feature_encoding: Literal['None', 'OneHotEncoder', 'OrdinalEncoder']="None"):
    """
    This function encodes a single feature using either 'None', 'OneHotEncoder', or 'OrdinalEncoder'.
    If 'None' is specified, the original feature is returned without encoding.
    If 'OneHotEncoder' is specified, the feature is one-hot encoded.
    If 'OrdinalEncoder' is specified, the feature is ordinally encoded.
    The function also supports sparse output for the OneHotEncoder. 
    The sparse output is controlled by the `sparse_output` parameter, which defaults to False.
    If `sparse_output` is True, the OneHotEncoder will return a sparse matrix.
    If `sparse_output` is False, it will return a dense array.
    This function is useful for preprocessing categorical features before training machine learning models.

    Args:
        feature1 (array-like): The input feature to be encoded.
        sparse_output (bool): If True, the OneHotEncoder will return a sparse matrix.
        feature_encoding (str): The encoding method to be applied. Options are 'None', 'OneHotEncoder', 'OrdinalEncoder'.

    Returns:
        feature1_encoded (array-like): The encoded feature.
    """
    if feature_encoding == 'None':
        feature1_encoded = feature1
    if feature_encoding == 'OrdinalEncoder':
        encoder = OrdinalEncoder()
        feature1_encoded = encoder.fit_transform(feature1.reshape(-1,1))
    if feature_encoding == 'OneHotEncoder':
        encoder = OneHotEncoder(sparse_output=sparse_output)
        feature1_encoded = encoder.fit_transform(feature1.reshape(-1,1))
    return feature1_encoded

def print_results(model, accuracy, f1, compile_time, params):
    """
    This function prints the results of a machine learning model evaluation.
    It displays the model name, accuracy score, F1 score, time taken for compilation, and model parameters.
    This is useful for summarizing the performance of different models in a consistent format.
    Args:
        model (str): The name of the machine learning model.
        accuracy (float): The accuracy score of the model.
        f1 (float): The F1 score of the model.
        compile_time (float): The time taken to compile the model, in seconds.
        params (dict): The parameters used for the model.
    Returns:
        None
    """
    print(f"{model} Model Accuracy score: {accuracy:.4f}")
    print(f"{model} Model F1 score: {f1:.4f}")
    print(f"Time taken for {model} Model (secs): {compile_time:.4f}")
    print(f"{model} Model Params: ", params)