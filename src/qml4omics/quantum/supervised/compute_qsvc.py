import time 
from typing import Literal

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval
import qml4omics.utils.qutils as qutils

# ====== Scikit-learn imports ======


# ====== Qiskit imports ======

#from qiskit.primitives import Sampler
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def compute_qsvc(X_train, X_test, y_train, y_test, args, model='QSVC', data_key = '',
                 C=1, gamma='scale', pegasos=False, encoding: Literal['ZZ', 'Z', 'P']="ZZ",
                 entanglement='linear', primitive = 'sampler', reps = 2, verbose=False):
    beg_time = time.time()
    
    
    # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(feature_map=encoding,
                                         feat_dimension=X_train.shape[1], 
                                         reps = reps,
                                         entanglement=entanglement)


    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(args,
                                                             primitive,
                                                             num_qubits=feature_map.num_qubits)
    
    print(f"Currently running a quantum support vector classifier (QSVC) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")
    
    if 'simulator' == args['backend']:
        fidelity = ComputeUncompute(sampler=prim)
    else:    
        # Need to instatiate a basic pass manager to store the chosen hardware backend
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)           
        fidelity = ComputeUncompute(sampler=prim, pass_manager=pm) #, num_virtual_qubits = feature_map.num_qubits )
    
    Qkernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    if pegasos == True:
        qsvc = PegasosQSVC(C=C, gamma=gamma, quantum_kernel=Qkernel)
    else:
        qsvc = QSVC(C=C, gamma=gamma, quantum_kernel=Qkernel)
        
    model_fit = qsvc.fit(X_train, y_train)
    # model_params = model_fit.get_params()
    hyperparameters = {'feature_map': feature_map.__class__.__name__,
                        'quantum_kernel': Qkernel.__class__.__name__,
                        'C': C,
                        'gamma': gamma,
                        }
    model_params = hyperparameters
    y_predicted = qsvc.predict(X_test) 

    if not isinstance(session, type(None)):
        session.close()

    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))
