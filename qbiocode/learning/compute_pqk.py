# ====== Base class imports ======
import time
import numpy as np
import os

# ====== Additional local imports ======
from qbiocode.evaluation.model_evaluation import modeleval
import qbiocode.utils.qutils as qutils
from sklearn.model_selection import GridSearchCV

# ====== Qiskit imports ======
from qiskit import QuantumCircuit

#from qiskit.primitives import Sampler
from functools import reduce
from qiskit.quantum_info import Pauli
from sklearn import svm

def compute_pqk(X_train, X_test, y_train, y_test, args, model='PQK', data_key = '', verbose=False,
                 encoding = 'Z', primitive = 'estimator', entanglement = 'linear', reps= 2):
    """
    This function generates quantum circuits, computes projections of the data onto these circuits,
    and evaluates the performance of a Support Vector Classifier (SVC) on the projected data.
    It uses a feature map to encode the data into quantum states and then measures the expectation values
    of Pauli operators to obtain the features. The SVC is trained on the projected training data and
    evaluated on the projected test data. The function returns evaluation metrics and model parameters.
    This function requires a quantum backend (simulator or real quantum hardware) for execution.
    It supports various configurations such as encoding methods, entanglement strategies, and repetitions
    of the feature map. The results are saved to files for training and test projections, which are reused
    if they already exist to avoid redundant computations.
    This function is part of the main quantum machine learning pipeline (QProfiler.py) and is intended for use in supervised learning tasks.
    It leverages quantum computing to enhance feature extraction and classification performance on complex datasets.
    The function returns the performance results, including accuracy, F1-score, AUC, runtime, as well as model parameters, and other relevant metrics.
    
    Args:
        X_train (np.ndarray): Training data features.
        X_test (np.ndarray): Test data features.
        y_train (np.ndarray): Training data labels.
        y_test (np.ndarray): Test data labels.
        args (dict): Arguments containing backend and other configurations.
        model (str): Model type, default is 'PQK'.
        data_key (str): Key for the dataset, default is ''.
        verbose (bool): If True, print additional information, default is False.
        encoding (str): Encoding method for the quantum circuit, default is 'Z'.
        primitive (str): Primitive type to use, default is 'estimator'.
        entanglement (str): Entanglement strategy, default is 'linear'.
        reps (int): Number of repetitions for the feature map, default is 2.

    Returns:
        modeleval (dict): A dictionary containing evaluation metrics and model parameters.
    """

    beg_time = time.time()
    feat_dimension = X_train.shape[1]

    if not os.path.exists( 'pqk_projections'):
        os.makedirs('pqk_projections')

    file_projection_train = os.path.join( 'pqk_projections', 'pqk_projection_' + data_key + '_train.npy')
    file_projection_test = os.path.join( 'pqk_projections', 'pqk_projection_' + data_key + '_test.npy')

    
    #  This function ensures that all multiplicative factors of data features inside single qubit gates are 1.0
    def data_map_func(x: np.ndarray) -> float:
        """
        Define a function map from R^n to R.

        Args:
            x: data

        Returns:
            float: the mapped value
        """
        coeff = x[0] / 2 if len(x) == 1 else reduce(lambda m, n: (m * n) / 2, x)
        return coeff
    
     # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(feature_map=encoding,
                                         feat_dimension=X_train.shape[1], 
                                         reps = reps,
                                         entanglement=entanglement,
                                         data_map_func = data_map_func)

    # Build quantum circuit
    circuit = QuantumCircuit(feature_map.num_qubits)
    circuit.compose(feature_map, inplace=True)
    num_qubits = circuit.num_qubits

    if (not os.path.exists( file_projection_train ) ) | (not os.path.exists( file_projection_test ) ):

        #  Generate the backend, session and primitive
        backend, session, prim = qutils.get_backend_session(args,
                                                                'estimator',
                                                                num_qubits=num_qubits)

        # Transpile
        if args['backend'] != 'simulator':
            circuit = qutils.transpile_circuit( circuit, opt_level=3, backend = backend, 
                                            PT = True, initial_layout = None)
        

        for f_tr in [file_projection_train, file_projection_test]:
            if not os.path.exists( f_tr ):
                projections = []
                if 'train' in f_tr:
                    dat = X_train.copy()
                else:
                    dat = X_test.copy()
                
                # Identity operator on all qubits
                id = 'I' * feat_dimension

                # We group all commuting observables
                # These groups are the Pauli X, Y and Z operators on individual qubits
                # Apply the circuit layout to the observable if mapped to device
                if args['backend'] != 'simulator':
                    observables_x =[]
                    observables_y =[]
                    observables_z =[]
                    for i in range(feat_dimension):
                        observables_x.append( Pauli(id[:i] + 'X' + id[(i + 1):]).apply_layout(circuit.layout, num_qubits=backend.num_qubits) )
                        observables_y.append( Pauli(id[:i] + 'Y' + id[(i + 1):]).apply_layout(circuit.layout, num_qubits=backend.num_qubits) )
                        observables_z.append( Pauli(id[:i] + 'Z' + id[(i + 1):]).apply_layout(circuit.layout, num_qubits=backend.num_qubits) )
                else:
                    observables_x = [Pauli(id[:i] + 'X' + id[(i + 1):]) for i in range(feat_dimension)]
                    observables_y = [Pauli(id[:i] + 'Y' + id[(i + 1):]) for i in range(feat_dimension)]
                    observables_z = [Pauli(id[:i] + 'Z' + id[(i + 1):]) for i in range(feat_dimension)]
                    
                                                            
                # projections[i][j][k] will be the expectation value of the j-th Pauli operator (0: X, 1: Y, 2: Z)
                # of datapoint i on qubit k
                projections = []

                for i in range(len(dat)):
                    if i % 100 == 0:
                        print('at datapoint {}'.format(i))

                    # Get training sample 
                    parameters = dat[i]

                    # We define the primitive unified blocs (PUBs) consisting of the embedding circuit, 
                    # set of observables and the circuit parameters
                    pub_x = (circuit, observables_x, parameters)
                    pub_y = (circuit, observables_y, parameters)
                    pub_z = (circuit, observables_z, parameters)

                    job = prim.run([pub_x, pub_y, pub_z])
                    job_result_x = job.result()[0].data.evs
                    job_result_y = job.result()[1].data.evs
                    job_result_z = job.result()[2].data.evs

                    # Record <X>, <Y> and <Z> on all qubits for the current datapoint
                    projections.append([job_result_x, job_result_y, job_result_z])                                    
                np.save( f_tr, projections )

        if not isinstance(session, type(None)):
            session.close()

    # Load computed projections
    projections_train = np.load( file_projection_train )
    projections_train = np.array(projections_train).reshape(len(projections_train), -1)
    projections_test = np.load( file_projection_test )
    projections_test = np.array(projections_test).reshape(len(projections_test), -1)
    
    # Run SVC
    gridsearch_svc_args= {'C': [0.1, 1, 10, 100], 
                        'gamma': [0.001, 0.01, 0.1, 1],
                        'kernel': ['linear', 'rbf', 'poly','sigmoid']
                        }
    

    svc = svm.SVC(random_state=args['seed'])

    # Initialize GridSearchCV
    svc_random = GridSearchCV(estimator=svc, 
                                param_grid=gridsearch_svc_args, 
                                cv=5, 
                                n_jobs=-1)


    svc_random.fit(projections_train, y_train)
    y_predicted = svc_random.predict(projections_test)

    hyperparameters = {
                        'feature_map': feature_map.__class__.__name__,
                        'feature_map_reps': reps,
                        'entanglement' : entanglement,                        
                        'svc_best_params': svc_random.best_params_
                        # Add other hyperparameters as needed
                        }
    model_params = hyperparameters
    
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))