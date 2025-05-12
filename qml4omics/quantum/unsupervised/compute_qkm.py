import time 
import numpy as np
from typing import Literal

# ====== Additional local imports ======
from qml4omics.evaluation.model_evalutation import modeleval

# ====== Scikit-learn imports ======

from sklearn.model_selection import GridSearchCV
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score

# ====== Qiskit imports ======

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_algorithms.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap, PauliFeatureMap
from qiskit_aer import AerSimulator
from qiskit_algorithms.utils import algorithm_globals
from qiskit.primitives import Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC, PegasosQSVC
from qiskit_ibm_runtime import QiskitRuntimeService, Session, EstimatorV2 as Estimator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def compute_qkm(X_train, y_train,  X_test, y_test, verbose=False, model='QK-Means',
                encoding: Literal['ZZ', 'Z', 'P']="ZZ",):
    beg_time = time.time()   
    service = QiskitRuntimeService(channel="ibm_quantum")   
    
    feature_map = None

    if encoding == 'ZZ' :
        feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
    if encoding == 'Z': 
        feature_map = ZFeatureMap(feature_dimension=X_train.shape[1], reps=2)
    if encoding == 'P': 
        feature_map = PauliFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='linear')
    
    # sampler = Sampler() 
    # fidelity = ComputeUncompute(sampler=sampler)
    Qkernel = FidelityQuantumKernel(feature_map=feature_map)

    Q_train_matrix = Qkernel.evaluate(x_vec=X_train)
    qkm = SpectralClustering(2, affinity='precomputed')
    model_fit = qkm.fit_predict(Q_train_matrix)
    score = normalized_mutual_info_score(model_fit, y_train)
    return score
    # print(f"Model Accuracy score: {score:.4f}")
    # model_params = model_fit.__dict__
    # y_predicted = qkm.fit_predict(X_test) 
    # return(modeleval(y_test, y_predicted, beg_time, model_params, model=model, verbose=verbose))

