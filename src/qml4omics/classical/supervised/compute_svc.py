# ====== Base class imports ======

import time

# ====== Scikit-learn imports ======

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval
    
def compute_svc(X_train, X_test, y_train, y_test, args, model='SVC', data_key = '', C=1.0, kernel='rbf', 
                degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, 
                class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None):
        
    """ This function generates a model using a Support Vector Classifier (svc) method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed.
    """    
        
    beg_time = time.time()
    svc = OneVsOneClassifier(SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0, shrinking=shrinking, 
                                 probability=probability, tol=tol, cache_size=cache_size, class_weight=class_weight, 
                                 max_iter=max_iter, decision_function_shape=decision_function_shape, 
                                 break_ties=break_ties, random_state=random_state))
    # Fit the training datset
    model_fit = svc.fit(X_train, y_train)
    model_params = model_fit.get_params()
    # Validate the model in test dataset and calculate accuracy
    y_predicted = svc.predict(X_test) 
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))

def compute_svc_opt(X_train, X_test, y_train, y_test, args, verbose=False, cv=5, model='SVC',
                    C=[], gamma=[], kernel=[]):
        
    """ This function generates a model using a Support Vector Classifier (svc) method as implemented in scikit-learn 
    (https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVC.html). It takes in parameter
    arguments specified in the config.yaml file, but will use the default parameters specified above if none are passed. The
    combination of parameters that led to the best performance is saved and returned as best_params, which can then be used on similar
    datasets, without having to run the grid search.
    """   

    beg_time = time.time()
    params={'C': C,
            'gamma': gamma,
            'kernel': kernel
            }
    # Perform Grid Search to find the best parameters
    grid_search = GridSearchCV(SVC(), param_grid=params, cv=cv)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and use them to create the final SVC model
    best_params = grid_search.best_params_
    best_svc = SVC(**best_params)
    best_svc.fit(X_train, y_train)

    # Make predictions and calculate accuracy
    y_predicted = best_svc.predict(X_test)
    return(modeleval(y_test, y_predicted, beg_time, best_params, args, model=model, verbose=verbose))