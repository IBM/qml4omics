# ====== Base class imports ======
import os, json
import pandas as pd

# ====== Supervised CML functions imports ======
from qml4omics import compute_svc, compute_svc_opt
from qml4omics import compute_dt, compute_dt_opt
from qml4omics import compute_nb, compute_nb_opt
from qml4omics import compute_lr, compute_lr_opt
from qml4omics import compute_rf, compute_rf_opt
from qml4omics import compute_mlp, compute_mlp_opt

# ====== Supervised QML functions imports ======
from qml4omics import compute_qnn
from qml4omics import compute_qsvc
from qml4omics import compute_vqc
from qml4omics import compute_pqk

# ======= Parallelization =====
from joblib import Parallel, delayed


current_dir = os.getcwd()
compute_ml_dict = {'svc_opt' : compute_svc_opt,
                   'svc' : compute_svc,
                   'dt_opt' : compute_dt_opt,
                   'dt' : compute_dt,
                   'lr_opt' : compute_lr_opt,
                   'lr' : compute_lr,
                   'nb_opt' : compute_nb_opt,
                   'nb' : compute_nb,
                   'rf_opt' : compute_rf_opt,
                   'rf' : compute_rf,
                   'mlp_opt' : compute_mlp_opt,
                   'mlp' : compute_mlp,
                   'qsvc' : compute_qsvc,
                   'vqc' : compute_vqc,
                   'qnn' : compute_qnn,
                   'pqk' : compute_pqk
                   }

def model_run(X_train, X_test, y_train, y_test, data_key, args):
    """This function runs the ML methods, with or without a grid search, as specified in the config.yaml file.
    It returns a python dictionary contatining these results, which can then be parsed out. It is designed to run
    each of the ML methods in parallel, for each data set (this is done by calling the Parallel module in results below). 
    The arguments X_train, X_test, y_train, y_test are all passed in from the main script (qmlbench.py) as the input 
    datasets are processed, while the remaining arguments are passed from the config.yaml file. 
    
    Args:
        X_train: the training portion of the input data features, for that particular split. 
        X_test: the test portion of the input data features, for that particular split. 
        y_train: the training portion of the input data labels, for that particular split.
        y_test: the test portion of the input data labels, for that particular split. 
        args: additional arguments passed in from the config.yaml file
        
    Returns:
        model_total_result.to_dict(): a dictionary containing the model results for each method used. This dictionary can 
                                      readily be converted to a Pandas Dataframe.
    
    """
    
    # Run classical and quantum models
    n_jobs = len(args['model'])
    if 'n_jobs' in args.keys():
        n_jobs = min(args['n_jobs'], len(args['model']))
    
    grid_search = False    
    if 'grid_search' in args.keys():
        grid_search = args['grid_search']
    if grid_search:
        results = Parallel(n_jobs=n_jobs)(delayed(compute_ml_dict[method+ '_opt'])(X_train, X_test, y_train, y_test, args, model=method + '_opt',
                                                                                   cv = args['cross_validation'], 
                                                                                   **args['gridsearch_' + method + '_args'], 
                                                                                   verbose=False)
                                                                                   for method in args['model']) 
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(compute_ml_dict[method])(X_train, X_test, y_train, y_test, args, model=method, data_key = data_key,
                                                                           **args[method+'_args'], verbose=False)
                                                                           for method in args['model']) 
    
    model_total_result = pd.melt(pd.concat(results)).dropna()
    model_total_result['i'] = 0
    model_total_result = model_total_result.pivot(columns="variable", values="value", index="i")
    return model_total_result.to_dict()

