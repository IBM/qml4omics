# ====== Base class imports ======
import time
import numpy as np
import os

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval
import qml4omics.utils.qutils as qutils
from sklearn.model_selection import GridSearchCV

# ====== Qiskit imports ======
from qiskit import QuantumCircuit

#from qiskit.primitives import Sampler
from functools import reduce
from qiskit.quantum_info import Pauli
from sklearn import svm

def compute_pqk(X_train, X_test, y_train, y_test, args, model='PQK', data_key = '', verbose=False,
                 encoding = 'Z', primitive = 'estimator', entanglement = 'linear', reps= 2):
    beg_time = time.time()
    feat_dimension = X_train.shape[1]

    if not os.path.exists('./pqk'):
        os.mkdir('./pqk')

    file_projection_train = './pqk/pqk_projection_' + data_key + '_train.npy'
    file_projection_test = './pqk/pqk_projection_' + data_key + '_test.npy'

    
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

    print(f"Currently running a projected quantum kernel (PQK) method on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")
    
    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))