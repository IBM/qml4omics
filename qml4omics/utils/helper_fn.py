# ====== Base class imports ======

import time
from typing import Literal

# ====== Scikit-learn imports ======

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

def scaler_fn(X, scaling: Literal['None', 'StandardScaler', 'MinMaxScaler']="None"):
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
    print(f"{model} Model Accuracy score: {accuracy:.4f}")
    print(f"{model} Model F1 score: {f1:.4f}")
    print(f"Time taken for {model} Model (secs): {compile_time:.4f}")
    print(f"{model} Model Params: ", params)