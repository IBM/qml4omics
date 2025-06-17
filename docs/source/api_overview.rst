API Overview
============
Quick overview of the :ref:`methods` and :ref:`datasets` available in QML4Omics.

.. _methods:

Methods
-------
Depending on the underlying foundations in qml4omics can be....

Embeddings
^^^^^^^^^^
Collection of common embeddings  (:mod:`qml4omics.embeddings`) functionalities.

.. autosummary::
    ~qml4omics.embeddings.embed.get_embeddings


Evaluation
^^^^^^^^^^
The :mod:`qml4omics.evaluation` submodule of qml4omics computes the evaluation metrics for the input dataset and the models.

Data Evaluation
""""""""""""""""
Depending on the underlying mathematical foundations, they can be classified into the following categories: (i)..

.. autosummary::
    ~qml4omics.evaluation.dataset_evaluation.evaluate

Model Evaluation
""""""""""""""""

.. autosummary::
    ~qml4omics.evaluation.model_evaluation.modeleval


Model Computation 
^^^^^^^^^^^^^^^^^
qml4omics brings together a number of established machine learning model both from classical (:mod:`qml4omics.classical`)  and quantum (:mod:`qml4omics.quantum`).
Multiple models can be run via the following 

.. autosummary::
    ~qml4omics.evaluation.model_run.model_run

Classical Models
""""""""""""""""

Classical model....

.. autosummary::
    ~qml4omics.classical.supervised.compute_dt 
    ~qml4omics.classical.supervised.compute_lr 
    ~qml4omics.classical.supervised.compute_mlp 
    ~qml4omics.classical.supervised.compute_nb 
    ~qml4omics.classical.supervised.compute_rf 
    ~qml4omics.classical.supervised.compute_svc 

Quantum Models
""""""""""""""

Quantum model....


.. autosummary::
    ~qml4omics.quantum.supervised.compute_qnn.compute_qnn
    ~qml4omics.quantum.supervised.compute_qsvc.compute_qsvc
    ~qml4omics.quantum.supervised.compute_vqc.compute_vqc
    ~qml4omics.quantum.supervised.compute_pqk.compute_pqk



Visualisation
^^^^^^^^^^^^^
The plotting module (:mod:`qml4omics.visualization`) enables the user to visualise the data and provides out-of-the-box plots for some
of the metrics.

.. autosummary::
    ~qml4omics.visualization.visualize_correlation.compute_results_correlation
    ~qml4omics.visualization.visualize_correlation.plot_results_correlation
    
.. _datasets:

Datasets
-------- 
qml4omics provides... 

References
^^^^^^^^^^

