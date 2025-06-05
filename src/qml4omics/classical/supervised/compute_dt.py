# ====== Base class imports ======

import time

# ====== Scikit-learn imports ======

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval

# ====== Begin functions ======

def compute_dt(X_train, X_test, y_train, y_test, args, verbose=False, model='Decision Tree', data_key = '',criterion='gini', splitter='best', 
               max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, 
               random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0, 
               monotonic_cst=None):
    
    """This function generates a model using a Decision Tree (dt) Classifier method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    """ 
    
    beg_time = time.time()
    dt = OneVsOneClassifier(DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, 
                                                   min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, 
                                                   min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, 
                                                   random_state=random_state, max_leaf_nodes=max_leaf_nodes, 
                                                   min_impurity_decrease=min_impurity_decrease, class_weight=class_weight, 
                                                   ccp_alpha=ccp_alpha, monotonic_cst=monotonic_cst))
    # Fit the training datset
    model_fit = dt.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = dt.predict(X_test) 
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))

def compute_dt_opt(X_train, X_test, y_train, y_test, args, verbose=False, model='Decision Tree', cv=5, 
                   criterion=[], max_depth=[], min_samples_split=[], min_samples_leaf=[], max_features=[]):
    
    """This function also generates a model using a Decision Tree (dt) Classifier method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html). The difference here is that
    this function runs a grid search. The range of the grid search for each parameter is specified in the config.yaml file. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.
    """ 
    
    beg_time = time.time()
    params = {'criterion': criterion,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'max_features': max_features
              }
    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final Decision Tree model
    best_params = grid_search.best_params_
    best_dt = DecisionTreeClassifier(**best_params)
    best_dt.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_dt.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose))
