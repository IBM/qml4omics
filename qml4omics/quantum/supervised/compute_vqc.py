# ====== Base class imports ======
import time
from typing import Literal

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval
import qml4omics.utils.qutils as qutils

# ====== Qiskit imports ======
from qiskit_machine_learning.algorithms.classifiers import VQC
#from qiskit.primitives import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def compute_vqc(X_train, X_test, y_train, y_test, args, verbose=False, model='VQC', data_key = '',
                local_optimizer: Literal['COBYLA', 'L_BFGS_B', 'GradientDescent']="COBYLA", maxiter=100,
                encoding = 'Z', entanglement = 'linear', reps= 2,primitive = 'sampler', ansatz_type='amp'):
    beg_time = time.time()
     # choose a method for mapping your features onto the circuit
    feature_map, _ = qutils.get_feature_map(feature_map=encoding,
                                         feat_dimension=X_train.shape[1], 
                                         reps = reps,
                                         entanglement=entanglement)

    # get ansatz
    ansatz= qutils.get_ansatz( ansatz_type=ansatz_type, feat_dimension = feature_map.num_qubits, reps=reps, entanglement=entanglement)


    #  Generate the backend, session and primitive
    backend, session, prim = qutils.get_backend_session(args,
                                                             primitive,
                                                             num_qubits=feature_map.num_qubits)
        
    # Get Optimizer        
    optimizer = qutils.get_optimizer( local_optimizer, max_iter=maxiter)
    
     # instantiate the primitive
    if 'simulator' == args['backend']:
        vqc= VQC(sampler=prim, feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
    else:
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        vqc= VQC(sampler=prim, feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, pass_manager=pm)

    print(f"Currently running a variational quantum classifer (VQC) on this dataset.")
    print(f"The number of qubits in your circuit is: {feature_map.num_qubits}")
    print(f"The number of parameters in your circuit is: {feature_map.num_parameters}")
    
    # fit classifier to data
    model_fit = vqc.fit(X_train, y_train)
    hyperparameters = {
                        'feature_map': feature_map.__class__.__name__,
                        'ansatz': ansatz.__class__.__name__,
                        'optimizer': optimizer.__class__.__name__,
                        'optimizer_params': optimizer.settings,
                        # Add other hyperparameters as needed
                        }
    model_params = hyperparameters
    y_predicted = vqc.predict(X_test)

    if not isinstance(session, type(None)):
        session.close()

    return(modeleval(y_test, y_predicted, beg_time, model_params, args, model=model, verbose=verbose))