# ====== Base class imports ======

import time
import numpy as np

# ====== Scikit-learn imports ======

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval

# ====== Begin functions ======

def compute_mlp(X_train, X_test, y_train, y_test, args, verbose=False, model='Multi-layer Perceptron', data_key = '',
                hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', 
                learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=10000, shuffle=True, 
                random_state=None, tol=0.0001, warm_start=False, momentum=0.9, nesterovs_momentum=True, 
                early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
        
    """This function generates a model using a Multi-layer Perceptron (mlp), a neural network, method as implemented in scikit-learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    """  
    
    beg_time = time.time()
    mlp = OneVsOneClassifier(MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, 
                                           batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, 
                                           power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, 
                                           warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, 
                                           early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, 
                                           beta_2=beta_2, epsilon=epsilon, n_iter_no_change=n_iter_no_change, max_fun=max_fun))
    # Fit the training datset
    model_fit = mlp.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = mlp.predict(X_test) 
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))

def compute_mlp_opt(X_train, X_test, y_train, y_test, args, verbose=False, cv=5, model='Multi-layer Perceptron',
                    hidden_layer_sizes= [], activation = [], max_iter= [],
                    solver = [], alpha = [], learning_rate= []):
        
    """This function also generates a model using a Multi-layer Perceptron (mlp), a neural network, as implemented in scikit-learn 
    (https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html). The difference here is that
    this function runs a grid search. The range of the grid search for each parameter is specified in the config.yaml file. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.
    """   
    
    beg_time = time.time()
    params={'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation, 
            'max_iter': max_iter,
            'solver': solver,
            'alpha': alpha,
            'learning_rate': learning_rate,
            }
    
    # Pemlporm Grid Search to find the best parameters
    grid_search = GridSearchCV(MLPClassifier(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final model
    best_params = grid_search.best_params_
    best_mlp = MLPClassifier(**best_params)
    best_mlp.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_mlp.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose))
