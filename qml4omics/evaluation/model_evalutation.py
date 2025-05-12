# ====== Base class imports ======

import time
from typing import Literal
import pandas as pd

# ====== Scikit-learn imports ======

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from qml4omics.utils.helper_fn import print_results

def modeleval(y_test, y_predicted, beg_time, params, args, model:str, verbose = True, average='weighted'):
    auc = roc_auc_score(y_test, y_predicted)
    accuracy = accuracy_score(y_test, y_predicted, normalize=True)
    f1 = f1_score(y_test, y_predicted, average=average)
    compile_time = time.time() - beg_time
    params = params
    if verbose==True:
        print_results(model, accuracy, f1, compile_time, params)
    
    if args['grid_search'] == True: 
        return pd.DataFrame({'y_test_' + model: [y_test], 
                         'y_predicted_' + model: [y_predicted],
                         'results_' + model: [{'model':model,'accuracy': accuracy, 'f1_score': f1,'time': compile_time, 'auc': auc, 'BestParams_GridSearch': params}]})
    else: 
        return pd.DataFrame({'y_test_' + model: [y_test], 
                    'y_predicted_' + model: [y_predicted],
                    'results_' + model: [{'model':model,'accuracy': accuracy, 'f1_score': f1,'time': compile_time, 'auc': auc, 'Model_Parameters': params}]})