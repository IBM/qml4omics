# ====== Base class imports ======

import time
import numpy as np

# ====== Scikit-learn imports ======

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval

# ====== Begin functions ======
    
def compute_lr(X_train, X_test, y_train, y_test, args, model='Logistic Regression', data_key = '',
                   penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, 
                   class_weight=None, random_state=None, solver='saga', max_iter=10000, multi_class='deprecated', 
                   verbose=False, warm_start=False, n_jobs=None, l1_ratio=None):
    
    """This function generates a model using a Logistic Regression (lr) method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    """    
    
    beg_time = time.time()
    logres = OneVsOneClassifier(LogisticRegression(penalty=penalty, dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, 
                                                   intercept_scaling=intercept_scaling, class_weight=class_weight, random_state=random_state, 
                                                   solver=solver, max_iter=max_iter, multi_class=multi_class,
                                                   warm_start=warm_start, n_jobs=n_jobs, l1_ratio=l1_ratio))
    # Fit the training datset
    model_fit = logres.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = logres.predict(X_test) 
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))

def compute_lr_opt(X_train, X_test, y_train, y_test, args, model='Logistic Regression', cv=5,
                       penalty=[], C=[], 
                       solver=[], verbose=False, max_iter=[]):
    
    """This function also generates a model using a Logistic Regression (lr) method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html). The difference here is that
    this function runs a grid search. The range of the grid search for each parameter is specified in the config.yaml file. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.
    """  
    
    beg_time = time.time()
    params = {'penalty': penalty,
              'C': C,
              'solver':solver,
              'max_iter':max_iter
              }
    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(LogisticRegression(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final Decision Tree model
    best_params = grid_search.best_params_
    best_logres = LogisticRegression(**best_params)
    best_logres.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_logres.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose))