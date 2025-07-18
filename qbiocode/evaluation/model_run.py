# ====== Base class imports ======
import os, json
import pandas as pd

# ====== Supervised CML functions imports ======
from qbiocode import compute_svc, compute_svc_opt
from qbiocode import compute_dt, compute_dt_opt
from qbiocode import compute_nb, compute_nb_opt
from qbiocode import compute_lr, compute_lr_opt
from qbiocode import compute_rf, compute_rf_opt
from qbiocode import compute_mlp, compute_mlp_opt

# ====== Supervised QML functions imports ======
from qbiocode import compute_qnn
from qbiocode import compute_qsvc
from qbiocode import compute_vqc
from qbiocode import compute_pqk

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
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training labels.
        y_test (pd.Series): Testing labels.
        data_key (str): Key for the dataset being processed.
        args (dict): Dictionary containing configuration parameters, including:
            - model: List of models to run.
            - n_jobs: Number of parallel jobs to run.
            - grid_search: Boolean indicating whether to perform grid search.
            - cross_validation: Cross-validation strategy.
            - gridsearch_<model>_args: Arguments for grid search for each model.
            - <model>_args: Additional arguments for each model.
        
    Returns:
        model_total_result (dict): A dictionary containing the results of the models run, with keys as model names and values as their respective results.
        This dictionary can readily be converted to a Pandas Dataframe, as seen in the 'ModelResults.csv' files that are produced in the results directory
        when the main profiler is run (qbiocode-profiler.py).
    
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

