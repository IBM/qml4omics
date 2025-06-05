# ====== Base class imports ======

import time
import numpy as np

# ====== Scikit-learn imports ======

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval

# ====== Begin functions ======

def compute_rf(X_train, X_test, y_train, y_test, args, verbose=False, model='Random Forest', data_key = '',
               n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
               min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, 
               bootstrap=True, oob_score=False, n_jobs=None, random_state=None, warm_start=False, 
               class_weight=None, ccp_alpha=0.0, max_samples=None, monotonic_cst=None):
        
    """ This function generates a model using a Random Forest (rf) Classifier method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    """    
    
    beg_time = time.time()
    rf = OneVsOneClassifier(RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, 
                                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                                   min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                                   max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, 
                                                   bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, 
                                                   warm_start=warm_start, class_weight=class_weight, 
                                                   ccp_alpha=ccp_alpha, max_samples=max_samples, monotonic_cst=monotonic_cst))
    # Fit the training datset
    model_fit = rf.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = rf.predict(X_test) 
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))

def compute_rf_opt(X_train, X_test, y_train, y_test, args, verbose=False, cv=5, model='Random Forest',
                   bootstrap= [], max_depth= [], max_features= [],
                   min_samples_leaf= [], min_samples_split= [], n_estimators= []):
    
    """ This function also generates a model using a Random Forest (rf) Classifier method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html). The difference here is that
    this function runs a grid search. The range of the grid search for each parameter is specified in the config.yaml file. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.
    """  
    
    beg_time = time.time()
    params={'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap
            }
    
    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final model
    best_params = grid_search.best_params_
    best_rf = RandomForestClassifier(**best_params)
    best_rf.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_rf.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose))